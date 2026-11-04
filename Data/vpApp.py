"""
vpApp -- simple VisPy + PySide6 data viewer.

Layout:
    +------------------------------------------+
    | [Select folder]  file.parquet   top bar  |  <- LAYOUT.top_bar_frac of height
    +--------+------------------------------+--+
    |[Fl][Co]|                              |  |  <- side bar tabs: files / columns
    |  file  ||          graph grid           |+ |  <- right "+" adds a column
    |  tree  ||    (vplot Axes in one canvas) |  |
    |        +------------------------------+--+
    |        |              +               |  |  <- bottom "+" adds a row
    +--------+------------------------------+--+

Every number/color/name that defines the structure lives in LayoutConfig below.
The selected data folder is remembered between runs with QSettings.
Double-clicking a .parquet file in the tree loads it into parts.state.df (polars),
and its column names show up (searchable) in the side bar's Columns tab.
Right-click a graph to pick its x / y columns, labels, line names and colors,
or drag column names from the Columns tab onto a graph to add them as y lines.
Left-drag zooms to a box, Backspace undoes a zoom, Ctrl+Z / Ctrl+Y undo / redo
graph and file changes. 3D scatter graphs rotate on left drag instead.
The "Advanced" switch in the top right shows the features in
LayoutConfig.advanced_features (graph types, copy graph / y columns, custom colors).
"""

import copy
import json
import math
import os
import re
import sys
from dataclasses import asdict, dataclass, field, fields, replace
from typing import NamedTuple

import numpy as np
import polars as pl
from PySide6 import QtCore, QtGui, QtWidgets
from vispy import app as vispy_app
from vispy import scene

from FSLib import colscript
from FSLib import vplot as vp

vispy_app.use_app("pyside6")


# ---------------------------------------------------------------------------
# Structure config -- the one place that defines how the window is laid out
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LayoutConfig:
    window_title: str = "FS Data Viewer"
    window_size: tuple[int, int] = (1400, 900)    # initial (w, h) px

    top_bar_frac: float = 0.03                    # fraction of window height
    side_bar_frac: float = 0.10                   # initial fraction of width
    side_bar_min_px: int = 80                     # drag limits
    side_bar_max_frac: float = 0.50
    side_bar_grows_on_resize: bool = False        # False: main area takes extra width
    live_side_bar_resize: bool = False            # False: drag shows a guide line and
                                                  # the graphs resize once on release
    require_confirm_on_graph_delete: bool = False

    splitter_handle_px: int = 6
    side_tab_px: int = 26                         # height of the Files/Columns buttons
    files_tab_text: str = "Files"
    columns_tab_text: str = "Columns"
    column_search_placeholder: str = "Search columns..."
    tree_indent_px: int = 12
    tree_file_filters: tuple[str, ...] = ("*.parquet",)   # () shows every file
    tree_hide_empty_folders: bool = True          # hide folders with no data file inside
    tree_refresh_text: str = "\u27f3 Refresh"
    tree_refresh_tooltip: str = "Re-read the data folder (it isn't watched for changes)"

    select_button_text: str = "Select data folder"
    file_menu_text: str = "File"
    load_template_text: str = "Load template..."
    load_template_folder_text: str = "Load template folder..."
    replace_template_text: str = "Replace tabs with template..."
    replace_template_folder_text: str = "Replace tabs with template folder..."
    save_template_text: str = "Save template\tCtrl+S"
    save_template_as_text: str = "Save template as..."
    save_all_templates_text: str = "Save all templates..."

    # Tabs: each holds its own grid of graphs. A tab saves to one template file
    # (JSON); a folder of templates loads as one tab per file.
    templates_dir: str = "templates"              # default folder, next to vpApp.py
    template_suffix: str = ".json"
    template_format: str = "vpApp-tab"            # "format" key written to each file
    new_tab_name: str = "Tab {n}"
    add_tab_tooltip: str = "Add a tab"
    rename_tab_text: str = "Tab name:"
    tab_bar_px: int = 28

    # Settings menu (top right)
    settings_text: str = "Settings"
    confirm_tab_delete_text: str = "Confirm tab deletion"
    confirm_graph_delete_text: str = "Confirm row/column removal"
    settings_confirm_tab_key: str = "confirm_tab_delete"
    settings_confirm_graph_key: str = "confirm_graph_delete"
    confirm_tab_delete_default: bool = True       # graphs: require_confirm_on_graph_delete
    data_folder_name: str = "fs-data"             # expected name (case-insensitive)
    data_subfolder_prefix: str = "FS"             # shown at top level of fs-data
    data_file_suffix: str = ".parquet"

    settings_org: str = "FormulaSlug"
    settings_app: str = "vpApp"
    settings_data_dir_key: str = "data_dir"

    toast_ms: int = 3000
    toast_margin_px: int = 12

    # canvas edge -> graphs (left, top, right, bottom). The right side leaves room for
    # an x tick label centred on the plot's right edge.
    graph_margins_px: tuple[int, int, int, int] = (6, 6, 24, 12)
    graph_spacing_px: int = 12                    # gap between neighbouring graphs
    # Crowding: with n graphs stacked vertically, the gaps between rows and the x label
    # font are scaled by max(crowd_min_scale, 1 / (1 + crowd_shrink * (n - 1)));
    # columns do the same for the gaps between columns and the y label font.
    crowd_shrink: float = 0.15
    crowd_min_scale: float = 0.6
    plus_strip_px: int = 18                       # thickness of the "+" strips

    # Graph text, in points (pixel sizes follow from vplot.SCREEN_DPI).
    title_fontsize: float = 14.4
    label_fontsize: float = 12.0
    tick_fontsize: float = 10.2
    legend_fontsize: float = 10.2
    font_scale_3d: float = 2.0                    # 3D tick/axis text vs the 2D sizes
    xtick_pad_px: float | None = None             # axis line -> tick labels; None = vplot default
    ytick_pad_px: float | None = None

    # Left drag on a graph draws a box and zooms to it; Backspace undoes the last zoom.
    right_click_slop_px: int = 4                  # right press -> release further than this
                                                  # doesn't open the menu
    print_fps: bool = True                       # print the graph canvas' FPS to the console
    graph_menu_configure_text: str = "Configure graph..."
    graph_menu_autoscale_text: str = "Autoscale"
    graph_menu_undo_zoom_text: str = "Undo zoom\tBackspace"
    graph_menu_clear_text: str = "Clear graph"
    graph_menu_remove_row_text: str = "Remove row"
    graph_menu_remove_col_text: str = "Remove column"
    graph_menu_copy_graph_text: str = "Copy graph to"
    graph_menu_copy_y_text: str = "Copy y columns to"
    default_x_column: str = "Time"                # x column picked for a new graph
    default_xlabel: str = "Time (s)"              # x label used with default_x_column
    undo_limit: int = 50                          # Ctrl+Z / Ctrl+Y steps kept (layout / file)

    # Graph types: key -> name shown in the Configure dialog.
    graph_kinds: tuple[tuple[str, str], ...] = (
        ("line", "Line"), ("scatter", "Scatter"), ("scatter3d", "3D scatter"))
    rolling_max: int = 1_000_000                  # largest rolling-mean window (rows)
    rolling_center: bool = True                   # centred window (no time lag) vs trailing
    rolling_tooltip: str = "Rolling mean over this many rows (1 = off). 2D graphs only."
    rolling_box_px: int = 88                      # width of the rolling-mean box
    # Display transform: the rest of an expression that starts with the column's
    # value, e.g. "/ (32767 * 180) + 2". Numbers only; combine columns in the script.
    expr_placeholder: str = "e.g. / 1000 + 2"
    expr_tooltip: str = ("Applied to the column when plotted: the value, then this.\n"
                         "e.g. \"/ (32767 * 180) + 2\" or \"* -1\". Numbers only.\n"
                         "Empty = unchanged.")
    expr_edit_px: int = 150                       # width of the transform boxes
    expr_error_color: str = "#d9534f"
    default_graph_kind: str = "scatter"           # type of a new graph
    # 3D view: perspective uses this field of view; isometric is orthographic (fov 0)
    # and snaps to the isometric angles when switched on.
    perspective_fov: float = 30.0
    iso_elevation: float = 35.264
    iso_azimuth: float = -45.0
    graph_menu_isometric_text: str = "Isometric view"
    view_3d_options: tuple[tuple[str, str], ...] = (("perspective", "Perspective"),
                                                    ("isometric", "Isometric"))
    default_marker_size: float = 1.0              # scatter point diameter, points
    marker_size_range: tuple[float, float] = (0.5, 30.0)

    # Advanced mode (switch in the top right). With it off, these features are hidden:
    #   graph_type    graph type / point size / z column in the Configure dialog
    #   copy_graph    "Copy graph to" in the graph menu
    #   copy_y        "Copy y columns to" in the graph menu
    #   custom_color  "Custom color..." (the 20 preset colors stay)
    #   script        the "Script" button (new columns from formulas, see FSLib/colscript.py)
    #   label_toggles show legend / show x label / show y label in the Configure dialog
    #   view_3d       perspective / isometric choice for 3D graphs (dialog and menu)
    #   links         the "Links" button (link graphs' x / y axes)
    advanced_features: tuple[str, ...] = ("graph_type", "copy_graph", "copy_y", "custom_color",
                                          "script", "label_toggles", "view_3d", "links")

    # Axis links: linked 2D graphs share that axis's range (zoom one, all follow;
    # autoscale fits all their data). Edited in the overlay the Links button opens.
    links_button_text: str = "Links"
    add_x_link_text: str = "+x link"
    add_y_link_text: str = "+y link"
    links_done_text: str = "Done"
    links_cancel_text: str = "Cancel"
    links_help_text: str = ("Select a link, then click graphs to add or remove them. "
                            "Done merges overlapping links; Esc cancels.")
    link_box_color: tuple[int, int, int, int] = (120, 190, 255, 70)      # RGBA, checked graph
    link_box_border: tuple[int, int, int, int] = (60, 140, 230, 200)
    link_free_border: tuple[int, int, int, int] = (60, 140, 230, 60)     # unchecked 2D graph
    link_3d_color: tuple[int, int, int, int] = (128, 128, 128, 60)       # 3D: can't be linked
    link_box_gap_px: int = 6                      # space between boxes of adjacent graphs
    links_side_bar_px: int = 240                  # side bar is widened to this while editing
    advanced_text: str = "Advanced"
    settings_advanced_key: str = "advanced_mode"

    # Column script window
    script_button_text: str = "Script"
    script_window_title: str = "Column script"
    script_window_size: tuple[int, int] = (900, 560)
    script_columns_frac: float = 0.3              # width share of the column list
    script_font: str = "Consolas"
    script_font_pt: float = 10.5
    script_placeholder: str = ("# one new column per line, e.g.\n"
                               "power = voltage * current\n"
                               "energy = integral(power, Time)\n"
                               "accel = derivative(speed, Time)")
    script_save_text: str = "Save"
    script_save_close_text: str = "Save and close"
    script_copied_text: str = "Copied {name!r}"
    script_ok_text: str = "Saved. {count} script column(s)."
    script_unsaved_text: str = "Discard unsaved script changes?"
    script_error_color: str = "#d9534f"
    settings_script_key: str = "column_script"
    switch_size_px: tuple[int, int] = (34, 18)
    switch_on_color: str = "#5865f2"
    switch_off_color: str = "#4e5058"
    switch_knob_color: str = "#ffffff"
    column_mime_type: str = "application/x-fs-columns"   # column-list drag payload
    confirm_title: str = "Confirm"
    confirm_text: str = "Confirm"
    cancel_text: str = "Cancel"
    graph_dialog_title: str = "Configure graph"
    graph_dialog_size: tuple[int, int] = (768, 480)
    add_line_text: str = "+ Add y line"
    line_label_placeholder: str = "legend label (defaults to column)"
    custom_color_text: str = "Custom color..."
    swatch_px: int = 18
    # matplotlib's tab20, in its usual (dark, light) pairs.
    line_colors: tuple[str, ...] = (
        "#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c", "#98df8a",
        "#d62728", "#ff9896", "#9467bd", "#c5b0d5", "#8c564b", "#c49c94",
        "#e377c2", "#f7b6d2", "#7f7f7f", "#c7c7c7", "#bcbd22", "#dbdb8d",
        "#17becf", "#9edae5")

    top_bar_color: str = "#2b2d31"
    side_bar_color: str = "#313338"
    handle_color: str = "#1e1f22"
    graph_bgcolor: str = "white"                  # vplot draws black axes/text
    plus_color: str = "#2b2d31"
    plus_hover_color: str = "#404249"
    text_color: str = "#dbdee1"
    toast_color: str = "#c9822b"
    toast_text_color: str = "#1e1f22"
    tab_color: str = "#2b2d31"
    tab_checked_color: str = "#404249"
    input_color: str = "#1e1f22"


LAYOUT = LayoutConfig()


@dataclass
class LineSpec:
    column: str
    label: str
    color: str
    rolling: int = 1                              # rolling-mean window in rows; 1 = off (2D only)
    expr: str = ""                                # display transform, e.g. "/ 1000 + 2"


@dataclass
class GraphSpec:
    """What a graph shows; edited through the right-click dialog."""
    x: str | None = None
    lines: list[LineSpec] = field(default_factory=list)
    xlabel: str = ""
    ylabel: str = ""
    title: str = ""
    kind: str = LAYOUT.default_graph_kind         # a key of LayoutConfig.graph_kinds
    marker_size: float = LAYOUT.default_marker_size   # scatter point diameter (points)
    z: str | None = None                          # 3D scatter only
    zlabel: str = ""
    legend: bool = True                           # legend below (2D) / inside (3D) the graph
    show_xlabel: bool = True                      # 2D: hiding a label frees its space
    show_ylabel: bool = True
    projection: str = "perspective"               # 3D: "perspective" | "isometric"
    x_expr: str = ""                              # display transforms of x / z
    z_expr: str = ""


def is_3d(kind: str) -> bool:
    return kind == "scatter3d"


@dataclass(eq=False)                              # compare graphs by identity
class Graph:
    ax: "vp.Axes | vp.Axes3D | None"
    cell: scene.widgets.Grid                      # the grid cell the axes lives in
    spec: GraphSpec = field(default_factory=GraphSpec)


@dataclass(eq=False)
class AxisLink:
    """Graphs whose x (or y) axis shows the same range."""
    axis: str                                     # "x" | "y"
    graphs: list[Graph] = field(default_factory=list)


@dataclass
class GraphGrid:
    """One SceneCanvas holding a rows x cols grid of graphs. Each graph's cell is
    placed by layout_cells() (VisPy's Grid ignores its spacing setting)."""
    canvas: scene.SceneCanvas
    cfg: LayoutConfig
    cells: list[list[Graph]] = field(default_factory=list)
    zoom_history: list[Graph] = field(default_factory=list)   # graph of each zoom, oldest first
    rubber_band: QtWidgets.QRubberBand | None = None          # box-zoom outline over the canvas
    links: list[AxisLink] = field(default_factory=list)
    syncing: bool = False                         # set while sync_links moves cameras


@dataclass(eq=False)
class Tab:
    """One tab: its own graph grid (canvas + "+" strips), and the template file it
    was loaded from / saved to, if any."""
    name: str
    graphs: GraphGrid
    area: QtWidgets.QWidget                       # page in Tabs.stack
    path: str | None = None


@dataclass
class Tabs:
    bar: QtWidgets.QTabBar
    stack: QtWidgets.QStackedWidget
    items: list[Tab] = field(default_factory=list)   # in tab-bar order

    def current(self) -> Tab:
        return self.items[self.bar.currentIndex()]


class TabSnapshot(NamedTuple):
    name: str
    path: str | None
    specs: list[list[GraphSpec]]                  # by grid position
    links: tuple = ()                             # ((axis, ((row, col), ...)), ...)


class Snapshot(NamedTuple):
    """What Ctrl+Z restores: the loaded file, the column script, and every tab
    (name, template file, graph specs)."""
    df_path: str | None
    script: str
    tabs: list[TabSnapshot]
    current: int                                  # selected tab


@dataclass
class AppState:
    """Things that change while the app runs."""
    df: pl.DataFrame | None = None                       # file + script columns
    base_df: pl.DataFrame | None = None                  # the file as read
    df_path: str | None = None
    script: str = ""                                     # column script (colscript)
    undo: list[Snapshot] = field(default_factory=list)   # oldest first
    redo: list[Snapshot] = field(default_factory=list)   # oldest first
    advanced: bool = False                               # advanced-mode switch
    confirm_tab_delete: bool = True                      # Settings menu switches
    confirm_graph_delete: bool = True
    script_window: QtWidgets.QWidget | None = None       # open ScriptWindow, if any
    link_edit: "LinkEdit | None" = None                  # set while the links overlay is open


def feature_on(state: AppState, name: str, cfg: LayoutConfig) -> bool:
    """False for an advanced-only feature (cfg.advanced_features) while advanced mode is off."""
    return state.advanced or name not in cfg.advanced_features


class SideBar(NamedTuple):
    """Left bar: Files/Columns tab buttons over a stack of pages. The links page
    replaces both (tab buttons hidden) while axis links are being edited."""
    frame: QtWidgets.QFrame
    files_button: QtWidgets.QPushButton
    columns_button: QtWidgets.QPushButton
    stack: QtWidgets.QStackedWidget
    tree: QtWidgets.QTreeView
    column_search: QtWidgets.QLineEdit
    column_list: QtWidgets.QListWidget
    links: "LinksPage"


class LinksPage(NamedTuple):
    done: QtWidgets.QCheckBox
    cancel: QtWidgets.QPushButton
    add_x: QtWidgets.QPushButton
    add_y: QtWidgets.QPushButton
    list: QtWidgets.QListWidget


FILES_PAGE, COLUMNS_PAGE, LINKS_PAGE = 0, 1, 2


class AppParts(NamedTuple):
    """Handles to the pieces of the window, so other code can fill them in."""
    window: QtWidgets.QMainWindow
    top_bar: QtWidgets.QFrame
    side: SideBar
    splitter: QtWidgets.QSplitter
    tabs: Tabs
    file_button: QtWidgets.QPushButton           # File menu (templates, data folder)
    file_label: QtWidgets.QLabel
    script_button: QtWidgets.QPushButton
    links_button: QtWidgets.QPushButton
    state: AppState

    @property
    def graphs(self) -> GraphGrid:
        """The selected tab's graphs."""
        return self.tabs.current().graphs


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def simpleTimeCol (df):
    stepSize = 60/5035
    ser = pl.Series(np.arange(0,df.height*stepSize, stepSize)).alias("Time")
    return ser.slice(0, df.height)


def read (a:str):
    df = pl.read_parquet(a).fill_null(strategy="forward").fill_null(strategy="backward")
    if "Time" in df.columns:
        return df
    elif "Time_ms" in df.columns:
        return df.insert_column(0, (df["Time_ms"] / 1000).alias("Time"))
    else:
        return df.insert_column(0, simpleTimeCol(df))


def load_data_file(parts: AppParts, path: str | None, cfg: LayoutConfig,
                   undoable: bool = True) -> bool:
    """Read `path` (None unloads) and redraw every graph. Returns False if it failed."""
    if path is None:
        df = None
    else:
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            df = read(path)
        except (pl.exceptions.PolarsError, OSError) as e:
            show_toast(parts, f"Could not read {os.path.basename(path)}: {e}", cfg)
            return False
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
    if undoable:
        push_undo(parts, cfg)
    parts.state.base_df, parts.state.df_path = df, path
    if df is None:
        parts.file_label.setText("No file loaded")
        parts.file_label.setToolTip("")
    else:
        parts.file_label.setText(os.path.basename(path))
        parts.file_label.setToolTip(f"{path}\n{df.height:,} rows x {df.width} columns")
    errors = apply_script(parts, cfg)
    if errors:
        show_toast(parts, f"Column script: {len(errors)} line(s) skipped ({errors[0]})", cfg)
    return True


def apply_script(parts: AppParts, cfg: LayoutConfig) -> list[colscript.ScriptError]:
    """Rebuild state.df = file + script columns, refresh the column lists and redraw.
    Returns the errors of the script lines that were skipped."""
    state, errors = parts.state, []
    state.df = state.base_df
    if state.base_df is not None and state.script.strip():
        state.df, errors = colscript.run(state.script, state.base_df)
    fill_column_list(parts.side.column_list, parts.side.column_search, state.df)
    redraw_all(parts, cfg)
    return errors


def set_script(parts: AppParts, text: str, cfg: LayoutConfig,
               undoable: bool = True) -> list[colscript.ScriptError]:
    """Replace the column script (remembered between runs) and re-apply it."""
    if undoable and text != parts.state.script:
        push_undo(parts, cfg)
    parts.state.script = text
    _settings(cfg).setValue(cfg.settings_script_key, text)
    return apply_script(parts, cfg)


# ---------------------------------------------------------------------------
# Undo (Ctrl+Z) / redo (Ctrl+Y): graph layout, graph specs and file selection
# ---------------------------------------------------------------------------
def grid_specs(graphs: GraphGrid) -> list[list[GraphSpec]]:
    return [[copy.deepcopy(g.spec) for g in row] for row in graphs.cells]


def snapshot(parts: AppParts) -> Snapshot:
    tabs = [TabSnapshot(t.name, t.path, grid_specs(t.graphs), link_positions(t.graphs))
            for t in parts.tabs.items]
    return Snapshot(parts.state.df_path, parts.state.script, tabs,
                    parts.tabs.bar.currentIndex())


def push_undo(parts: AppParts, cfg: LayoutConfig) -> None:
    """Call before a change, so Ctrl+Z can come back to the current state.
    A new change drops whatever Ctrl+Y could have redone."""
    undo = parts.state.undo
    undo.append(snapshot(parts))
    del undo[:-cfg.undo_limit]
    parts.state.redo.clear()


def restore_snapshot(parts: AppParts, snap: Snapshot, cfg: LayoutConfig) -> bool:
    if snap.df_path != parts.state.df_path:
        old_script = parts.state.script
        parts.state.script = snap.script          # so the reload applies the right script
        if not load_data_file(parts, snap.df_path, cfg, undoable=False):
            parts.state.script = old_script
            return False
        _settings(cfg).setValue(cfg.settings_script_key, snap.script)
    elif snap.script != parts.state.script:
        set_script(parts, snap.script, cfg, undoable=False)
    refresh_script_window(parts)
    set_tabs(parts, snap.tabs, cfg)
    parts.tabs.bar.setCurrentIndex(min(snap.current, len(parts.tabs.items) - 1))
    return True


def step_history(parts: AppParts, back: bool, cfg: LayoutConfig) -> None:
    """Ctrl+Z (back=True) / Ctrl+Y: move one snapshot between the undo and redo stacks.
    Ignored while the links overlay is open (finish or cancel it first)."""
    if parts.state.link_edit is not None:
        return
    source, dest = ((parts.state.undo, parts.state.redo) if back
                    else (parts.state.redo, parts.state.undo))
    if not source:
        return
    current = snapshot(parts)
    if restore_snapshot(parts, source[-1], cfg):
        source.pop()
        dest.append(current)
        del dest[:-cfg.undo_limit]


# ---------------------------------------------------------------------------
# Data folder persistence
# ---------------------------------------------------------------------------
def _settings(cfg: LayoutConfig) -> QtCore.QSettings:
    return QtCore.QSettings(cfg.settings_org, cfg.settings_app)


def load_data_dir(cfg: LayoutConfig) -> str | None:
    path = _settings(cfg).value(cfg.settings_data_dir_key, None)
    return path if path and os.path.isdir(path) else None


def save_data_dir(path: str, cfg: LayoutConfig) -> None:
    _settings(cfg).setValue(cfg.settings_data_dir_key, path)


def is_data_folder(path: str, cfg: LayoutConfig) -> bool:
    return os.path.basename(os.path.normpath(path)).lower() == cfg.data_folder_name.lower()


# ---------------------------------------------------------------------------
# File tree
# ---------------------------------------------------------------------------
class TopLevelFilter(QtCore.QSortFilterProxyModel):
    """Inside `root_path`: hides top-level entries that aren't folders starting with
    `prefix` (None shows all), and folders with no `suffix` file anywhere inside
    (when `hide_empty`). The folder checks are cached until clear_cache()."""

    def __init__(self, suffix: str, hide_empty: bool):
        super().__init__()
        self.root_path = ""
        self.prefix = None
        self.suffix, self.hide_empty = suffix.lower(), hide_empty
        self._has_data: dict[str, bool] = {}

    def clear_cache(self):
        self._has_data.clear()

    def has_data(self, folder: str) -> bool:
        """True if a data file is somewhere below `folder` (stops at the first one)."""
        if folder not in self._has_data:
            self._has_data[folder] = self._scan(folder)
        return self._has_data[folder]

    def _scan(self, folder: str) -> bool:
        stack = [folder]
        while stack:
            try:
                with os.scandir(stack.pop()) as entries:
                    for entry in entries:
                        if entry.is_file() and entry.name.lower().endswith(self.suffix):
                            return True
                        if entry.is_dir(follow_symlinks=False):
                            stack.append(entry.path)
            except OSError:                       # unreadable folder: skip it
                continue
        return False

    def filterAcceptsRow(self, row, parent):
        if not self.root_path:                    # no data folder yet: drives etc.
            return True
        model = self.sourceModel()
        index = model.index(row, 0, parent)
        path = QtCore.QDir.cleanPath(model.filePath(index))
        if not path.startswith(self.root_path + "/"):
            return True                           # the root's ancestors
        if self.prefix is not None and QtCore.QDir.cleanPath(
                model.filePath(parent)) == self.root_path:
            if not (model.isDir(index) and model.fileName(index).startswith(self.prefix)):
                return False
        if self.hide_empty and model.isDir(index):
            return self.has_data(model.filePath(index))
        return True


def build_fs_model(cfg: LayoutConfig) -> QtWidgets.QFileSystemModel:
    model = QtWidgets.QFileSystemModel()
    # The folder isn't watched; the Refresh button re-reads it.
    model.setOption(QtWidgets.QFileSystemModel.DontWatchForChanges, True)
    # AllDirs: the name filters below apply to files only, not folder names.
    model.setFilter(QtCore.QDir.AllDirs | QtCore.QDir.Files | QtCore.QDir.Drives
                    | QtCore.QDir.NoDotAndDotDot)
    if cfg.tree_file_filters:
        model.setNameFilters(list(cfg.tree_file_filters))
        model.setNameFilterDisables(False)        # hide, rather than grey out
    model.sort(0)                                 # folders first, then by name
    return model


def build_file_tree(cfg: LayoutConfig) -> QtWidgets.QTreeView:
    model = build_fs_model(cfg)
    proxy = TopLevelFilter(cfg.data_file_suffix, cfg.tree_hide_empty_folders)
    proxy.setSourceModel(model)

    tree = QtWidgets.QTreeView()
    tree.setModel(proxy)
    tree.setHeaderHidden(True)
    for col in range(1, model.columnCount()):     # keep only the name column
        tree.hideColumn(col)
    tree.setIndentation(cfg.tree_indent_px)
    tree.setFrameShape(QtWidgets.QFrame.NoFrame)
    tree.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
    tree.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
    tree.setExpandsOnDoubleClick(False)           # clicks handled in on_tree_*clicked
    return tree


def expanded_paths(tree: QtWidgets.QTreeView) -> list[str]:
    """Paths of the expanded folders under the tree's root, parents first."""
    proxy, out = tree.model(), []
    pending = [tree.rootIndex()]
    while pending:
        parent = pending.pop()
        for row in range(proxy.rowCount(parent)):
            index = proxy.index(row, 0, parent)
            if tree.isExpanded(index):
                out.append(tree_path(tree, index))
                pending.append(index)
    return out


def refresh_tree(tree: QtWidgets.QTreeView, cfg: LayoutConfig) -> None:
    """Re-read the data folder: a fresh file model, folder checks redone, and the
    folders that were open opened again as they load."""
    proxy = tree.model()
    root = proxy.root_path
    if not root:
        return
    reopen = expanded_paths(tree)
    old = proxy.sourceModel()
    model = build_fs_model(cfg)
    proxy.clear_cache()
    proxy.setSourceModel(model)
    old.deleteLater()
    for col in range(1, model.columnCount()):
        tree.hideColumn(col)
    model.setRootPath(root)
    tree.setRootIndex(proxy.mapFromSource(model.index(root)))

    def expand_loaded(folder):
        for path in reopen:
            if QtCore.QDir.cleanPath(os.path.dirname(path)) == QtCore.QDir.cleanPath(folder):
                tree.setExpanded(proxy.mapFromSource(model.index(path)), True)

    model.directoryLoaded.connect(expand_loaded)


def tree_path(tree: QtWidgets.QTreeView, index: QtCore.QModelIndex) -> str:
    proxy = tree.model()
    return proxy.sourceModel().filePath(proxy.mapToSource(index))


def on_tree_clicked(tree: QtWidgets.QTreeView, index: QtCore.QModelIndex) -> None:
    """Folders open/close on a single click."""
    if os.path.isdir(tree_path(tree, index)):
        tree.setExpanded(index, not tree.isExpanded(index))


def on_tree_double_clicked(parts: AppParts, index: QtCore.QModelIndex,
                           cfg: LayoutConfig) -> None:
    """Data files get loaded on a double click."""
    path = tree_path(parts.side.tree, index)
    if os.path.isfile(path) and path.lower().endswith(cfg.data_file_suffix):
        load_data_file(parts, path, cfg)


def set_tree_root(tree: QtWidgets.QTreeView, path: str, cfg: LayoutConfig) -> None:
    proxy = tree.model()
    model = proxy.sourceModel()
    new_filter = hasattr(proxy, "beginFilterChange")   # Qt >= 6.10 replaces invalidateFilter
    if new_filter:
        proxy.beginFilterChange()
    proxy.root_path = QtCore.QDir.cleanPath(path)
    proxy.prefix = cfg.data_subfolder_prefix if is_data_folder(path, cfg) else None
    proxy.clear_cache()
    if new_filter:
        proxy.endFilterChange()
    else:
        proxy.invalidateFilter()
    model.setRootPath(path)
    tree.setRootIndex(proxy.mapFromSource(model.index(path)))


# ---------------------------------------------------------------------------
# Column list
# ---------------------------------------------------------------------------
class ColumnList(QtWidgets.QListWidget):
    """Column names; dragging carries the selected names (one per line) as `mime_type`.
    `entries` holds every (name, tooltip); the list shows the ones matching the search."""

    def __init__(self, mime_type: str):
        super().__init__()
        self.mime_type = mime_type
        self.entries: list[tuple[str, str]] = []

    def mimeData(self, items):
        mime = QtCore.QMimeData()
        mime.setData(self.mime_type, "\n".join(i.text() for i in items).encode())
        return mime


def dropped_columns(mime: QtCore.QMimeData, cfg: LayoutConfig) -> list[str]:
    if not mime.hasFormat(cfg.column_mime_type):
        return []
    return bytes(mime.data(cfg.column_mime_type)).decode().split("\n")


def build_column_page(cfg: LayoutConfig) -> tuple[QtWidgets.QWidget, QtWidgets.QLineEdit,
                                                  QtWidgets.QListWidget]:
    page = QtWidgets.QWidget()
    search = QtWidgets.QLineEdit()
    search.setPlaceholderText(cfg.column_search_placeholder)
    search.setClearButtonEnabled(True)
    search.setStyleSheet(f"background-color: {cfg.input_color}; border: none; padding: 4px;")
    columns = ColumnList(cfg.column_mime_type)
    columns.setFrameShape(QtWidgets.QFrame.NoFrame)
    columns.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
    columns.setDragDropMode(QtWidgets.QAbstractItemView.DragOnly)   # drop onto a graph
    search.textChanged.connect(lambda text: filter_column_list(columns, text))

    layout = QtWidgets.QVBoxLayout(page)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(0)
    layout.addWidget(search)
    layout.addWidget(columns)
    return page, search, columns


def filter_column_list(columns: ColumnList, text: str) -> None:
    """Show the entries matching `text` (case-insensitive substring; empty = all).
    The list is rebuilt rather than rows hidden: with hidden rows QListView kept
    the old scroll offset and applied it on the next click, jumping down."""
    needle = text.strip().lower()
    columns.clear()
    for name, tooltip in columns.entries:
        if needle in name.lower():
            item = QtWidgets.QListWidgetItem(name)
            item.setToolTip(tooltip)
            columns.addItem(item)
    columns.scrollToTop()                         # clear() keeps the old scroll offset


def fill_column_list(columns: ColumnList, search: QtWidgets.QLineEdit,
                     df: pl.DataFrame | None) -> None:
    columns.entries = [(name, str(dtype))
                       for name, dtype in ({} if df is None else df.schema).items()]
    filter_column_list(columns, search.text())


# ---------------------------------------------------------------------------
# Graph grid
# ---------------------------------------------------------------------------
def apply_vplot_params(cfg: LayoutConfig) -> None:
    """Push the graph text settings into vplot (read when each Axes is created)."""
    vp.rcParams.update({
        "axes.titlesize": cfg.title_fontsize,
        "axes.labelsize": cfg.label_fontsize,
        "tick.labelsize": cfg.tick_fontsize,
        "legend.fontsize": cfg.legend_fontsize,
        "xtick.pad": cfg.xtick_pad_px,
        "ytick.pad": cfg.ytick_pad_px,
    })


def new_cell(graphs: GraphGrid) -> scene.widgets.Grid:
    """An empty cell for one graph's axes; layout_cells() sizes and places it."""
    return scene.widgets.Grid(parent=graphs.canvas.scene)


def add_graph(graphs: GraphGrid) -> Graph:
    graph = Graph(None, new_cell(graphs))
    attach_axes(graphs, graph, three_d=False)
    return graph


def attach_axes(graphs: GraphGrid, graph: Graph, three_d: bool) -> None:
    """Fill graph.cell with 2D axes (left drag box-zooms) or 3D axes (left drag
    rotates, Shift + drag pans, wheel zooms)."""
    if three_d:
        graph.ax = vp.Axes3D(None, graph.cell)
        return
    graph.ax = vp.Axes(None, graph.cell, box_zoom=True)
    graph.ax.view.camera.on_box_zoom = lambda camera: graphs.zoom_history.append(graph)
    graph.ax.view.camera.on_box_drag = lambda p1, p2: show_rubber_band(graphs, p1, p2)
    graph.ax.view.scene.transform.changed.connect(lambda event: sync_links(graphs, graph))


def match_axes(graphs: GraphGrid, graph: Graph) -> None:
    """Swap the graph's axes when its spec switches between 2D and 3D. A cell can't
    be emptied and reused, so a fresh cell takes the old one's place."""
    three_d = is_3d(graph.spec.kind)
    if isinstance(graph.ax, vp.Axes3D) == three_d:
        return
    graph.cell.parent = None
    graph.cell = new_cell(graphs)
    attach_axes(graphs, graph, three_d)
    graphs.zoom_history[:] = [g for g in graphs.zoom_history if g is not graph]
    layout_cells(graphs)


def has_box_zoom(graph: Graph) -> bool:
    return isinstance(graph.ax.view.camera, vp.BoxZoomCamera)


def show_rubber_band(graphs: GraphGrid, p1, p2) -> None:
    """Box-zoom outline drawn by Qt, so dragging doesn't redraw the graphs."""
    band = graphs.rubber_band
    if p1 is None:
        band.hide()
        return
    band.setGeometry(QtCore.QRect(QtCore.QPoint(*map(round, p1)),
                                  QtCore.QPoint(*map(round, p2))).normalized())
    band.show()


def add_graph_row(graphs: GraphGrid) -> None:
    graphs.cells.append([add_graph(graphs) for _ in graphs.cells[0]])
    refresh_layout(graphs)


def add_graph_col(graphs: GraphGrid) -> None:
    for cells_row in graphs.cells:
        cells_row.append(add_graph(graphs))
    refresh_layout(graphs)


def remove_graph_row(graphs: GraphGrid, row: int) -> None:
    drop_graphs(graphs, removed=graphs.cells.pop(row))


def remove_graph_col(graphs: GraphGrid, col: int) -> None:
    drop_graphs(graphs, removed=[cells_row.pop(col) for cells_row in graphs.cells])


def drop_graphs(graphs: GraphGrid, removed: list[Graph]) -> None:
    """Take `removed` off the canvas; the rest move up/left to fill the space."""
    for graph in removed:
        graph.cell.parent = None
    graphs.zoom_history[:] = [g for g in graphs.zoom_history if g not in removed]
    for link in graphs.links:
        link.graphs = [g for g in link.graphs if g not in removed]
    graphs.links = [link for link in graphs.links if len(link.graphs) >= 2]
    refresh_layout(graphs)


def crowd_scale(n: int, cfg: LayoutConfig) -> float:
    """Shrink factor for gaps/label text with n graphs side by side (1 for n = 1)."""
    return max(cfg.crowd_min_scale, 1 / (1 + cfg.crowd_shrink * (n - 1)))


def layout_cells(graphs: GraphGrid) -> None:
    """Size and place every graph's cell: equal cells inside the margins, with gaps
    that shrink as rows (vertical gap) or columns (horizontal gap) are added."""
    cfg = graphs.cfg
    w, h = graphs.canvas.size
    left, top, right, bottom = cfg.graph_margins_px
    rows, cols = len(graphs.cells), len(graphs.cells[0])
    gap_x = cfg.graph_spacing_px * crowd_scale(cols, cfg)
    gap_y = cfg.graph_spacing_px * crowd_scale(rows, cfg)
    cell_w = max((w - left - right - gap_x * (cols - 1)) / cols, 1)
    cell_h = max((h - top - bottom - gap_y * (rows - 1)) / rows, 1)
    for r, cells_row in enumerate(graphs.cells):
        for c, graph in enumerate(cells_row):
            graph.cell.pos = (left + c * (cell_w + gap_x), top + r * (cell_h + gap_y))
            graph.cell.size = (cell_w, cell_h)
    graphs.canvas.update()


def refresh_layout(graphs: GraphGrid) -> None:
    """After rows/columns change: re-place the cells and re-size the label text."""
    layout_cells(graphs)
    for graph in all_graphs(graphs):
        style_axes(graphs, graph)


def style_axes(graphs: GraphGrid, graph: Graph) -> None:
    """Titles, axis labels (text, size, shown/hidden) and the 3D view, from the spec.
    2D label text shrinks with crowding: x labels with the number of rows, y labels
    with the number of columns. 3D text is font_scale_3d times the 2D sizes."""
    cfg, ax, spec = graphs.cfg, graph.ax, graph.spec
    ax.set_title(spec.title)
    if isinstance(ax, vp.Axes3D):
        camera = ax.view.camera
        isometric = spec.projection == "isometric"
        if isometric and camera.fov != 0:         # switching to isometric: snap the angles
            camera.elevation, camera.azimuth = cfg.iso_elevation, cfg.iso_azimuth
        camera.fov = 0 if isometric else cfg.perspective_fov
        big = cfg.font_scale_3d
        if isometric:
            # VisPy shrinks 3D text by its depth under perspective (it's sized in clip
            # space, before the divide by w); orthographic has no such divide. Divide
            # by the perspective depth at the centre so both views match.
            big /= 0.5 * camera.scale_factor / math.tan(math.radians(cfg.perspective_fov) / 2)
        for setter, text in ((ax.set_xlabel, spec.xlabel), (ax.set_ylabel, spec.ylabel),
                             (ax.set_zlabel, spec.zlabel)):
            setter(text, fontsize=cfg.label_fontsize * big)
        ax.tick_params(labelsize=cfg.tick_fontsize * big)
        return
    rows, cols = len(graphs.cells), len(graphs.cells[0])
    ax.set_xlabel(spec.xlabel, fontsize=cfg.label_fontsize * crowd_scale(rows, cfg))
    ax.set_ylabel(spec.ylabel, fontsize=cfg.label_fontsize * crowd_scale(cols, cfg))
    ax.set_label_visible("x", spec.show_xlabel)
    ax.set_label_visible("y", spec.show_ylabel)


def set_grid_shape(graphs: GraphGrid, rows: int, cols: int) -> None:
    """Add/remove rows and columns at the bottom/right until the grid is rows x cols."""
    while len(graphs.cells) > rows:
        remove_graph_row(graphs, len(graphs.cells) - 1)
    while len(graphs.cells[0]) > cols:
        remove_graph_col(graphs, len(graphs.cells[0]) - 1)
    while len(graphs.cells) < rows:
        add_graph_row(graphs)
    while len(graphs.cells[0]) < cols:
        add_graph_col(graphs)


def graph_position(graphs: GraphGrid, graph: Graph) -> tuple[int, int]:
    for r, cells_row in enumerate(graphs.cells):
        if graph in cells_row:
            return r, cells_row.index(graph)
    raise ValueError("graph is not in the grid")


def all_graphs(graphs: GraphGrid) -> list[Graph]:
    return [g for row in graphs.cells for g in row]


def remember_zoom(graphs: GraphGrid, graph: Graph) -> None:
    """Save graph's current view so Backspace can come back to it."""
    graph.ax.view.camera.remember()
    graphs.zoom_history.append(graph)


def undo_zoom(graphs: GraphGrid) -> None:
    """Step back the most recent zoom, whichever graph it was on."""
    while graphs.zoom_history:
        graph = graphs.zoom_history.pop()
        if has_box_zoom(graph) and graph.ax.view.camera.undo_zoom():
            graphs.canvas.update()
            return


def build_graph_grid(cfg: LayoutConfig) -> GraphGrid:
    apply_vplot_params(cfg)
    # keys=None: VisPy's "interactive" keys close the canvas on Escape.
    canvas = scene.SceneCanvas(keys=None, bgcolor=cfg.graph_bgcolor, show=False,
                               dpi=vp.SCREEN_DPI)
    vp.skip_offscreen_swap(canvas)                # halves frame time on Intel GPUs
    graphs = GraphGrid(canvas, cfg,
                       rubber_band=QtWidgets.QRubberBand(QtWidgets.QRubberBand.Rectangle,
                                                         canvas.native))
    vp.keep_ticks_synced(canvas, lambda: [g.ax for g in all_graphs(graphs)
                                          if isinstance(g.ax, vp.Axes)])
    if cfg.print_fps:
        canvas.measure_fps()
    canvas.events.resize.connect(lambda event: layout_cells(graphs))
    graphs.cells.append([add_graph(graphs)])
    refresh_layout(graphs)
    return graphs


def draw_graph(graphs: GraphGrid, graph: Graph, df: pl.DataFrame | None,
               cfg: LayoutConfig) -> None:
    """Rebuild a graph from its spec; columns missing from df are skipped."""
    match_axes(graphs, graph)
    ax, spec = graph.ax, graph.spec
    if has_box_zoom(graph):
        # Saved zooms belong to the old data/limits (e.g. the empty graph's 0..1).
        ax.view.camera.clear_zoom_history()
    ax.cla()
    style_axes(graphs, graph)
    for line, data in plot_data(spec, df, cfg):
        if spec.kind == "line":
            ax.plot(*data, color=line.color, label=line.label)
        else:                                     # scatter size is an area, in points^2
            ax.scatter(*data, s=spec.marker_size ** 2, color=line.color, label=line.label)
    ax.autoscale()
    # 3D axes have no slot below them, so their legend sits inside the plot.
    # With the legend off, cla() above already removed it (the space below collapses).
    if spec.legend:
        ax.legend(loc="upper right" if is_3d(spec.kind) else "below",
                  fontsize=cfg.legend_fontsize)


def plot_data(spec: GraphSpec, df: pl.DataFrame | None, cfg: LayoutConfig):
    """(line, (x, y[, z]) arrays) for each line whose columns are all in df."""
    needed = [spec.x, spec.z] if is_3d(spec.kind) else [spec.x]
    if df is None or not all(c in df.columns for c in needed):
        return []
    x = transformed(df[spec.x], spec.x_expr)
    if is_3d(spec.kind):
        z = transformed(df[spec.z], spec.z_expr)
        return [(line, (x, y_values(df[line.column], line, 1, df, cfg), z))
                for line in spec.lines if line.column in df.columns]
    return [(line, (x, y_values(df[line.column], line, line.rolling, df, cfg)))
            for line in spec.lines if line.column in df.columns]


def y_values(column: pl.Series, line: LineSpec, rolling: int, df: pl.DataFrame,
             cfg: LayoutConfig) -> np.ndarray:
    """The column as plotted: a rolling mean over `rolling` rows when that's above 1
    (the ends average over the rows available, so no points go missing), then
    the line's display transform."""
    if rolling > 1:
        column = column.cast(pl.Float64).rolling_mean(
            window_size=rolling, min_samples=1, center=cfg.rolling_center)
    return transformed(column, line.expr)


def transformed(column: pl.Series, expr: str) -> np.ndarray:
    """The column with its display transform applied (see colscript.transform). An
    invalid transform (e.g. from an older template) leaves the column unchanged."""
    if not expr.strip():
        return column.to_numpy()
    values = column.cast(pl.Float64).to_numpy()
    try:
        return colscript.transform(expr, values)
    except colscript.ScriptError:
        return values


def redraw_all(parts: "AppParts", cfg: LayoutConfig) -> None:
    """Redraw every graph in every tab (e.g. after a new file or script)."""
    for tab in parts.tabs.items:
        redraw_grid(parts, tab.graphs, cfg)


def redraw_grid(parts: "AppParts", graphs: GraphGrid, cfg: LayoutConfig) -> None:
    for graph in all_graphs(graphs):
        draw_graph(graphs, graph, parts.state.df, cfg)
    fit_links(graphs)
    graphs.canvas.update()


# ---------------------------------------------------------------------------
# Axis links
# ---------------------------------------------------------------------------
def linkable(graph: Graph) -> bool:
    """Only 2D graphs (box-zoom cameras) can share an axis."""
    return graph.ax is not None and has_box_zoom(graph)


def sync_links(graphs: GraphGrid, source: Graph) -> None:
    """`source`'s view changed (zoom, Backspace, autoscale...): give the graphs
    linked to it the same range on the linked axis."""
    if graphs.syncing or not graphs.links or not linkable(source):
        return
    r = source.ax.view.camera.rect
    graphs.syncing = True
    try:
        for link in graphs.links:
            if source not in link.graphs:
                continue
            for graph in link.graphs:
                if graph is source or not linkable(graph):
                    continue
                camera = graph.ax.view.camera
                g = camera.rect
                new = ((r.left, g.bottom, r.width, g.height) if link.axis == "x"
                       else (g.left, r.bottom, g.width, r.height))
                if not np.allclose(new, (g.left, g.bottom, g.width, g.height)):
                    camera.rect = new
    finally:
        graphs.syncing = False


def fit_links(graphs: GraphGrid, only: Graph | None = None) -> None:
    """Fit each link's axis to all of its graphs' data (only links holding `only`,
    if given)."""
    for link in graphs.links:
        if only is not None and only not in link.graphs:
            continue
        members = [g for g in link.graphs if linkable(g)]
        i = 0 if link.axis == "x" else 1
        bounds = [b for g in members for b in (a.bounds() for a in g.ax._artists)
                  if b is not None]
        if not bounds:
            continue
        lo, hi = vp._pad(min(b[i][0] for b in bounds), max(b[i][1] for b in bounds))
        graphs.syncing = True
        try:
            for graph in members:
                camera = graph.ax.view.camera
                g = camera.rect
                camera.rect = ((lo, g.bottom, hi - lo, g.height) if link.axis == "x"
                               else (g.left, lo, g.width, hi - lo))
        finally:
            graphs.syncing = False


def link_positions(graphs: GraphGrid) -> tuple:
    """The links by grid position, for snapshots and templates."""
    return tuple((link.axis, tuple(graph_position(graphs, g) for g in link.graphs))
                 for link in graphs.links)


def links_from_positions(graphs: GraphGrid, positions) -> list[AxisLink]:
    rows, cols = len(graphs.cells), len(graphs.cells[0])
    links = []
    for axis, cells in positions:
        members = [graphs.cells[r][c] for r, c in cells if 0 <= r < rows and 0 <= c < cols]
        if axis in ("x", "y") and len(members) >= 2:
            links.append(AxisLink(axis, members))
    return links


def merged_links(graphs: GraphGrid, links: list[AxisLink]) -> list[AxisLink]:
    """Links of the same axis that share a graph become one; links with fewer than
    two graphs are dropped. Members are kept in grid order."""
    order = {g: i for i, g in enumerate(all_graphs(graphs))}
    out = []
    for axis in ("x", "y"):
        groups: list[set[Graph]] = []
        for link in links:
            if link.axis != axis:
                continue
            group = set(link.graphs)
            for other in [g for g in groups if g & group]:
                groups.remove(other)
                group |= other
            groups.append(group)
        out += [AxisLink(axis, sorted(group, key=order.__getitem__))
                for group in groups if len(group) >= 2]
    return out


def canvas_rect(widget: scene.widgets.Widget) -> tuple[float, float, float, float]:
    """(x, y, w, h) of a VisPy widget in canvas pixels."""
    x = y = 0.0
    node = widget
    while isinstance(node, scene.widgets.Widget):
        x, y = x + node.pos[0], y + node.pos[1]
        node = node.parent
    return x, y, widget.width, widget.height


def graph_at(graphs: GraphGrid, pos) -> Graph | None:
    for graph in all_graphs(graphs):
        x, y, w, h = canvas_rect(graph.cell)
        if x <= pos[0] < x + w and y <= pos[1] < y + h:
            return graph
    return None


def build_plus_strip(vertical: bool, cfg: LayoutConfig) -> QtWidgets.QPushButton:
    button = QtWidgets.QPushButton("+")
    button.setStyleSheet(
        f"QPushButton {{ background-color: {cfg.plus_color}; color: {cfg.text_color};"
        " border: none; font-weight: bold; }"
        f"QPushButton:hover {{ background-color: {cfg.plus_hover_color}; }}")
    if vertical:
        button.setFixedWidth(cfg.plus_strip_px)
        button.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Expanding)
        button.setToolTip("Add a column of graphs")
    else:
        button.setFixedHeight(cfg.plus_strip_px)
        button.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        button.setToolTip("Add a row of graphs")
    return button


def build_graph_area(graphs: GraphGrid, before_change, cfg: LayoutConfig) -> QtWidgets.QWidget:
    """Graph canvas with a "+" strip on the right (columns) and bottom (rows).
    before_change() is called before a strip adds graphs (for Ctrl+Z)."""
    area = QtWidgets.QWidget()
    layout = QtWidgets.QGridLayout(area)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(0)
    add_col = build_plus_strip(vertical=True, cfg=cfg)
    add_row = build_plus_strip(vertical=False, cfg=cfg)
    add_col.clicked.connect(lambda: (before_change(), add_graph_col(graphs)))
    add_row.clicked.connect(lambda: (before_change(), add_graph_row(graphs)))
    # Backspace only while the canvas has focus (clicking a graph gives it focus),
    # so it doesn't fire while typing in the search box.
    graphs.canvas.native.setFocusPolicy(QtCore.Qt.StrongFocus)
    undo = QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Backspace), graphs.canvas.native)
    undo.setContext(QtCore.Qt.WidgetShortcut)
    undo.activated.connect(lambda: undo_zoom(graphs))
    layout.addWidget(graphs.canvas.native, 0, 0)
    layout.addWidget(add_col, 0, 1)
    layout.addWidget(add_row, 1, 0)
    return area


# ---------------------------------------------------------------------------
# Right-click menu
# ---------------------------------------------------------------------------
def connect_graph_menu(parts: "AppParts", graphs: GraphGrid, cfg: LayoutConfig) -> None:
    """A right click (press and release without moving) opens the menu for the
    graph under the mouse."""
    canvas = graphs.canvas
    press = {}

    def on_press(event):
        if event.button == 2:
            press["pos"] = np.asarray(event.pos, dtype=float)

    def on_release(event):
        start = press.pop("pos", None)
        if event.button != 2 or start is None:
            return
        if np.hypot(*(np.asarray(event.pos, dtype=float) - start)) > cfg.right_click_slop_px:
            return
        graph = graph_at(graphs, event.pos)
        if graph is not None:
            # Deferred so the menu's event loop doesn't run inside VisPy's handler.
            QtCore.QTimer.singleShot(0, lambda: show_graph_menu(parts, graph, cfg))

    canvas.events.mouse_press.connect(on_press)
    canvas.events.mouse_release.connect(on_release)


def confirm(parent: QtWidgets.QWidget, text: str, cfg: LayoutConfig) -> bool:
    """Confirm / Cancel box; Enter confirms, Escape cancels."""
    box = QtWidgets.QMessageBox(parent)
    box.setWindowTitle(cfg.confirm_title)
    box.setText(text)
    yes = box.addButton(cfg.confirm_text, QtWidgets.QMessageBox.AcceptRole)
    no = box.addButton(cfg.cancel_text, QtWidgets.QMessageBox.RejectRole)
    box.setDefaultButton(yes)
    box.setEscapeButton(no)
    box.exec()
    return box.clickedButton() is yes


def show_graph_menu(parts: "AppParts", graph: Graph, cfg: LayoutConfig) -> None:
    graphs = parts.graphs
    row, col = graph_position(graphs, graph)

    def autoscale():
        if has_box_zoom(graph):
            remember_zoom(graphs, graph)
        graph.ax.autoscale()
        fit_links(graphs, only=graph)             # linked axes fit all their graphs
        graphs.canvas.update()

    def clear():
        set_spec(parts, graph, GraphSpec(), cfg)

    def confirmed(text):
        # The Settings menu switch (default: require_confirm_on_graph_delete).
        return not parts.state.confirm_graph_delete or confirm(parts.window, text, cfg)

    def remove_row():
        if confirmed(f"Remove row {row + 1} ({len(graphs.cells[0])} graphs)?"):
            push_undo(parts, cfg)
            remove_graph_row(graphs, row)

    def remove_col():
        if confirmed(f"Remove column {col + 1} ({len(graphs.cells)} graphs)?"):
            push_undo(parts, cfg)
            remove_graph_col(graphs, col)

    menu = QtWidgets.QMenu(parts.window)
    menu.addAction(cfg.graph_menu_configure_text, lambda: configure_graph(parts, graph, cfg))
    menu.addAction(cfg.graph_menu_autoscale_text, autoscale)
    menu.addAction(cfg.graph_menu_undo_zoom_text, lambda: undo_zoom(graphs)).setEnabled(
        bool(graphs.zoom_history))
    menu.addAction(cfg.graph_menu_clear_text, clear)
    if is_3d(graph.spec.kind) and feature_on(parts.state, "view_3d", cfg):
        isometric = graph.spec.projection == "isometric"
        iso = menu.addAction(cfg.graph_menu_isometric_text, lambda: set_spec(
            parts, graph,
            replace(graph.spec, projection="perspective" if isometric else "isometric"), cfg))
        iso.setCheckable(True)
        iso.setChecked(graph.spec.projection == "isometric")
    others = [g for g in all_graphs(graphs) if g is not graph]
    if feature_on(parts.state, "copy_graph", cfg):
        add_target_menu(menu, cfg.graph_menu_copy_graph_text, graphs, others,
                        lambda target: set_spec(parts, target, copy.deepcopy(graph.spec), cfg))
    if feature_on(parts.state, "copy_y", cfg):
        add_target_menu(menu, cfg.graph_menu_copy_y_text, graphs, others,
                        lambda target: add_y_columns(parts, target, graph.spec.lines, cfg),
                        enabled=bool(graph.spec.lines))
    menu.addSeparator()
    menu.addAction(cfg.graph_menu_remove_row_text, remove_row).setEnabled(len(graphs.cells) > 1)
    menu.addAction(cfg.graph_menu_remove_col_text, remove_col).setEnabled(
        len(graphs.cells[0]) > 1)
    menu.exec(QtGui.QCursor.pos())


def graph_name(graphs: GraphGrid, graph: Graph) -> str:
    row, col = graph_position(graphs, graph)
    title = graph.spec.title or ", ".join(line.label for line in graph.spec.lines) or "empty"
    return f"Row {row + 1}, column {col + 1}  ({title})"


def add_target_menu(menu: QtWidgets.QMenu, text: str, graphs: GraphGrid, targets: list[Graph],
                    action, enabled: bool = True) -> None:
    """Submenu `text` with one entry per graph in `targets`; picking one calls action(graph)."""
    sub = menu.addMenu(text)
    for target in targets:
        sub.addAction(graph_name(graphs, target), lambda t=target: action(t))
    sub.setEnabled(enabled and bool(targets))


def set_spec(parts: "AppParts", graph: Graph, spec: GraphSpec, cfg: LayoutConfig) -> None:
    """Give `graph` a new spec (undoable) and redraw it."""
    if spec == graph.spec:
        return
    push_undo(parts, cfg)
    graph.spec = spec
    draw_graph(parts.graphs, graph, parts.state.df, cfg)
    fit_links(parts.graphs, only=graph)
    parts.graphs.canvas.update()


def configure_graph(parts: "AppParts", graph: Graph, cfg: LayoutConfig) -> None:
    spec = run_graph_dialog(parts.window, graph.spec, parts.state.df,
                            lambda name: feature_on(parts.state, name, cfg), cfg)
    if spec is not None:
        set_spec(parts, graph, spec, cfg)


# ---------------------------------------------------------------------------
# Dragging columns from the side bar onto a graph
# ---------------------------------------------------------------------------
def default_x(df: pl.DataFrame | None, cfg: LayoutConfig) -> str | None:
    columns = numeric_columns(df)
    return cfg.default_x_column if cfg.default_x_column in columns else (
        columns[0] if columns else None)


def add_y_columns(parts: "AppParts", graph: Graph, columns: list[str | LineSpec],
                  cfg: LayoutConfig) -> None:
    """Add columns (names, or LineSpecs whose labels are kept) as y lines of `graph`,
    each with a color the graph doesn't use yet. Columns that are already shown or
    (with a file loaded) not numeric are skipped. An empty graph also gets the
    default x column and label."""
    df = parts.state.df
    numeric = set(numeric_columns(df))
    shown = {line.column for line in graph.spec.lines}
    spec = copy.deepcopy(graph.spec)
    for item in columns:
        # A copied line keeps its label, rolling mean and transform; only the color is new.
        line = LineSpec(item, item, "") if isinstance(item, str) else copy.copy(item)
        if line.column in shown or (df is not None and line.column not in numeric):
            continue
        shown.add(line.column)
        line.color = unused_line_color(cfg, spec.lines)
        spec.lines.append(line)
    if spec.x is None and spec.lines:
        spec.x = default_x(df, cfg) or cfg.default_x_column
        if spec.x == cfg.default_x_column and not spec.xlabel:
            spec.xlabel = cfg.default_xlabel
    set_spec(parts, graph, spec, cfg)


class ColumnDropFilter(QtCore.QObject):
    """Event filter on the graph canvas: accepts column drags over a graph."""

    def __init__(self, parts: "AppParts", graphs: GraphGrid, cfg: LayoutConfig):
        super().__init__(graphs.canvas.native)
        self.parts, self.graphs, self.cfg = parts, graphs, cfg

    def target(self, event) -> tuple[Graph | None, list[str]]:
        names = dropped_columns(event.mimeData(), self.cfg)
        pos = event.position()
        graph = graph_at(self.graphs, (pos.x(), pos.y())) if names else None
        return graph, names

    def eventFilter(self, obj, event):
        kind = event.type()
        if kind in (QtCore.QEvent.DragEnter, QtCore.QEvent.DragMove):
            graph, _ = self.target(event)
            event.setAccepted(graph is not None)
            if graph is not None:
                event.acceptProposedAction()
            return True
        if kind == QtCore.QEvent.Drop:
            graph, names = self.target(event)
            if graph is not None:
                event.acceptProposedAction()
                add_y_columns(self.parts, graph, names, self.cfg)
            return True
        return False


def connect_column_drops(parts: "AppParts", graphs: GraphGrid, cfg: LayoutConfig) -> None:
    native = graphs.canvas.native
    native.setAcceptDrops(True)
    native.installEventFilter(ColumnDropFilter(parts, graphs, cfg))   # parented to native


# ---------------------------------------------------------------------------
# Graph dialog
# ---------------------------------------------------------------------------
class LineRow(NamedTuple):
    widget: QtWidgets.QWidget
    column: QtWidgets.QComboBox
    label: QtWidgets.QLineEdit
    rolling: QtWidgets.QSpinBox
    transform: "TransformEdit"
    color: QtWidgets.QToolButton


class TransformEdit(NamedTuple):
    """One-line display transform, e.g. "/ (32767 * 180) + 2"."""
    widget: QtWidgets.QLineEdit

    def text(self) -> str:
        return self.widget.text().strip()


def numeric_columns(df: pl.DataFrame | None) -> list[str]:
    return [] if df is None else [n for n, t in df.schema.items() if t.is_numeric()]


def line_color(cfg: LayoutConfig, i: int) -> str:
    """i-th default line color: the dark half of tab20 first, then the light half."""
    order = cfg.line_colors[0::2] + cfg.line_colors[1::2]
    return order[i % len(order)]


def unused_line_color(cfg: LayoutConfig, lines: list[LineSpec]) -> str:
    """First default color no line in `lines` uses (cycles once all 20 are taken)."""
    used = {line.color.lower() for line in lines}
    for i in range(len(cfg.line_colors)):
        if line_color(cfg, i).lower() not in used:
            return line_color(cfg, i)
    return line_color(cfg, len(lines))


def build_column_combo(columns: list[str], current: str | None) -> QtWidgets.QComboBox:
    """Editable combo; typing filters the column names (substring match)."""
    combo = QtWidgets.QComboBox()
    combo.setEditable(True)
    combo.setInsertPolicy(QtWidgets.QComboBox.NoInsert)
    combo.addItems(columns)
    combo.completer().setFilterMode(QtCore.Qt.MatchContains)
    combo.completer().setCompletionMode(QtWidgets.QCompleter.PopupCompletion)
    combo.setCurrentIndex(combo.findText(current) if current else -1)
    combo.setEditText(current or "")
    return combo


def set_button_color(button: QtWidgets.QToolButton, color: str) -> None:
    button.setProperty("color", color)
    button.setStyleSheet(f"background-color: {color}; border: 1px solid #888;")
    button.setToolTip(color)


def build_color_button(color: str, custom: bool, cfg: LayoutConfig) -> QtWidgets.QToolButton:
    button = QtWidgets.QToolButton()
    button.setFixedSize(round(cfg.swatch_px * 1.5), cfg.swatch_px)
    set_button_color(button, color)
    button.clicked.connect(lambda: show_color_menu(button, custom, cfg))
    return button


def show_color_menu(button: QtWidgets.QToolButton, custom: bool, cfg: LayoutConfig) -> None:
    """Popup with the 20 preset swatches, plus a full color picker if `custom`."""
    menu = QtWidgets.QMenu(button)

    def choose(color):
        set_button_color(button, color)
        menu.close()

    swatches = QtWidgets.QWidget()
    grid = QtWidgets.QGridLayout(swatches)
    grid.setContentsMargins(4, 4, 4, 4)
    grid.setSpacing(2)
    for i, color in enumerate(cfg.line_colors):
        swatch = QtWidgets.QToolButton()
        swatch.setFixedSize(cfg.swatch_px, cfg.swatch_px)
        swatch.setStyleSheet(f"background-color: {color}; border: none;")
        swatch.setToolTip(color)
        swatch.clicked.connect(lambda checked=False, c=color: choose(c))
        grid.addWidget(swatch, i % 2, i // 2)     # dark shades on top, light below
    action = QtWidgets.QWidgetAction(menu)
    action.setDefaultWidget(swatches)
    menu.addAction(action)
    if custom:
        menu.addSeparator()
        menu.addAction(cfg.custom_color_text, lambda: pick_custom_color(button))
    menu.exec(button.mapToGlobal(QtCore.QPoint(0, button.height())))


def pick_custom_color(button: QtWidgets.QToolButton) -> None:
    color = QtWidgets.QColorDialog.getColor(QtGui.QColor(button.property("color")), button)
    if color.isValid():
        set_button_color(button, color.name())


def build_line_row(columns: list[str], line: LineSpec, on_remove, custom_color: bool,
                   cfg: LayoutConfig) -> LineRow:
    widget = QtWidgets.QWidget()
    combo = build_column_combo(columns, line.column)
    label = QtWidgets.QLineEdit(line.label)
    label.setPlaceholderText(cfg.line_label_placeholder)
    color = build_color_button(line.color, custom_color, cfg)
    rolling = QtWidgets.QSpinBox()
    rolling.setRange(1, cfg.rolling_max)
    rolling.setValue(line.rolling)
    rolling.setToolTip(cfg.rolling_tooltip)
    rolling.setFixedWidth(cfg.rolling_box_px)
    transform = build_transform(line.expr, columns, cfg)
    remove = QtWidgets.QToolButton()
    remove.setText("\u2715")
    remove.setToolTip("Remove line")

    # The label follows the column until the user types their own (clearing it
    # hands it back to the column).
    label.setProperty("auto", line.label in ("", line.column))
    label.textEdited.connect(lambda text: label.setProperty("auto", text == ""))
    combo.currentTextChanged.connect(
        lambda text: label.setText(text) if label.property("auto") else None)

    layout = QtWidgets.QHBoxLayout(widget)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.addWidget(combo, 3)
    layout.addWidget(label, 2)
    layout.addWidget(rolling)
    layout.addWidget(transform.widget)
    layout.addWidget(color)
    layout.addWidget(remove)
    row = LineRow(widget, combo, label, rolling, transform, color)
    remove.clicked.connect(lambda: on_remove(row))
    return row


def build_transform(expr: str, columns: list[str], cfg: LayoutConfig) -> TransformEdit:
    """Transform box; turns red (with the reason as its tooltip) while invalid."""
    edit = QtWidgets.QLineEdit(expr)
    edit.setPlaceholderText(cfg.expr_placeholder)
    edit.setFixedWidth(cfg.expr_edit_px)

    def check(text):
        error = colscript.check_transform(text)
        edit.setProperty("error", error)
        edit.setStyleSheet(f"border: 1px solid {cfg.expr_error_color};" if error else "")
        edit.setToolTip(f"{error}\n\n{cfg.expr_tooltip}" if error else cfg.expr_tooltip)

    edit.textChanged.connect(check)
    check(expr)
    return TransformEdit(edit)


def column_with_transform(combo: QtWidgets.QComboBox,
                          transform: TransformEdit) -> QtWidgets.QWidget:
    """A column picker with its transform box on the same line."""
    widget = QtWidgets.QWidget()
    layout = QtWidgets.QHBoxLayout(widget)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.addWidget(combo, 1)
    layout.addWidget(transform.widget)
    return widget


def build_kind_combo(kind: str, cfg: LayoutConfig) -> QtWidgets.QComboBox:
    combo = QtWidgets.QComboBox()
    for key, name in cfg.graph_kinds:
        combo.addItem(name, key)
    combo.setCurrentIndex(max(combo.findData(kind), 0))
    return combo


def build_marker_size(size: float, cfg: LayoutConfig) -> QtWidgets.QDoubleSpinBox:
    box = QtWidgets.QDoubleSpinBox()
    box.setRange(*cfg.marker_size_range)
    box.setSingleStep(0.5)
    box.setDecimals(1)
    box.setSuffix(" pt")
    box.setValue(size)
    return box


def run_graph_dialog(parent: QtWidgets.QWidget, spec: GraphSpec, df: pl.DataFrame | None,
                     feature, cfg: LayoutConfig) -> GraphSpec | None:
    """Modal editor for a graph's spec. Returns the new spec, or None on cancel.
    feature(name) says whether an advanced-mode feature is shown."""
    columns = numeric_columns(df)
    dialog = QtWidgets.QDialog(parent)
    dialog.setWindowTitle(cfg.graph_dialog_title)
    dialog.resize(*cfg.graph_dialog_size)

    title = QtWidgets.QLineEdit(spec.title)
    new_graph = spec.x is None and not spec.lines
    kind = build_kind_combo(spec.kind, cfg)
    marker_size = build_marker_size(spec.marker_size, cfg)
    x_combo = build_column_combo(columns, spec.x or default_x(df, cfg) or cfg.default_x_column)
    x_transform = build_transform(spec.x_expr, columns, cfg)
    x_row = column_with_transform(x_combo, x_transform)
    xlabel = QtWidgets.QLineEdit(spec.xlabel)
    if new_graph and not spec.xlabel and x_combo.currentText() == cfg.default_x_column:
        xlabel.setText(cfg.default_xlabel)
    ylabel = QtWidgets.QLineEdit(spec.ylabel)
    z_combo = build_column_combo(columns, spec.z)
    z_transform = build_transform(spec.z_expr, columns, cfg)
    z_row = column_with_transform(z_combo, z_transform)
    zlabel = QtWidgets.QLineEdit(spec.zlabel)
    view_3d = QtWidgets.QComboBox()
    for key, name in cfg.view_3d_options:
        view_3d.addItem(name, key)
    view_3d.setCurrentIndex(max(view_3d.findData(spec.projection), 0))
    toggles = QtWidgets.QWidget()                 # advanced: show legend / x label / y label
    toggles_layout = QtWidgets.QHBoxLayout(toggles)
    toggles_layout.setContentsMargins(0, 0, 0, 0)
    legend, show_x, show_y = (QtWidgets.QCheckBox(text) for text in
                              ("Show legend", "Show x label", "Show y label"))
    for box, on in ((legend, spec.legend), (show_x, spec.show_xlabel),
                    (show_y, spec.show_ylabel)):
        box.setChecked(on)
        toggles_layout.addWidget(box)
    toggles_layout.addStretch(1)
    # The layouts go on the dialog before anything is shown/hidden: setVisible(True)
    # on a widget without a parent opens it as its own window (they flashed up).
    layout = QtWidgets.QVBoxLayout(dialog)
    form = QtWidgets.QFormLayout()
    layout.addLayout(form)
    form.addRow("Title", title)
    form.addRow("Graph type", kind)
    form.addRow("Point size", marker_size)
    form.addRow("3D view", view_3d)
    form.addRow("X column", x_row)
    form.addRow("X label", xlabel)
    form.addRow("Y label", ylabel)
    form.addRow("Z column", z_row)
    form.addRow("Z label", zlabel)
    form.addRow("", toggles)

    def show_kind_rows():
        """Point size only for scatter, Z and the view only for 3D, x/y label toggles
        only for 2D; the type and the toggles are advanced-mode features."""
        key = kind.currentData()
        form.setRowVisible(kind, feature("graph_type"))
        form.setRowVisible(marker_size, key != "line")
        for widget in (z_row, zlabel):
            form.setRowVisible(widget, is_3d(key))
        form.setRowVisible(view_3d, is_3d(key) and feature("view_3d"))
        form.setRowVisible(toggles, feature("label_toggles"))
        show_x.setVisible(not is_3d(key))
        show_y.setVisible(not is_3d(key))
        for row in rows:                          # rolling mean is 2D only
            row.rolling.setVisible(not is_3d(key))

    rows: list[LineRow] = []
    rows_box = QtWidgets.QWidget()
    rows_layout = QtWidgets.QVBoxLayout(rows_box)
    rows_layout.addStretch(1)                     # rows are inserted above this
    scroll = QtWidgets.QScrollArea(dialog)
    scroll.setWidgetResizable(True)
    scroll.setWidget(rows_box)

    def remove_row(row):
        rows.remove(row)
        row.widget.deleteLater()

    def add_row(line):
        row = build_line_row(columns, line, remove_row, feature("custom_color"), cfg)
        rows_layout.insertWidget(len(rows), row.widget)
        rows.append(row)
        row.rolling.setVisible(not is_3d(kind.currentData()))

    for line in spec.lines:
        add_row(line)
    if not rows:
        add_row(LineSpec("", "", line_color(cfg, 0)))
    add_button = QtWidgets.QPushButton(cfg.add_line_text)
    add_button.clicked.connect(lambda: add_row(LineSpec("", "", line_color(cfg, len(rows)))))
    kind.currentIndexChanged.connect(show_kind_rows)
    show_kind_rows()

    buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok
                                         | QtWidgets.QDialogButtonBox.Cancel)
    def accept():
        """OK only once every transform is valid; otherwise say which isn't."""
        boxes = [("X", x_transform), ("Z", z_transform)] + [
            (r.column.currentText() or "a y line", r.transform) for r in rows]
        bad = [(name, t.widget.property("error")) for name, t in boxes
               if t.widget.property("error") and t.widget.isVisibleTo(dialog)]
        if bad:
            QtWidgets.QMessageBox.warning(dialog, cfg.graph_dialog_title, "\n".join(
                f"{name} transform: {error}" for name, error in bad))
            return
        dialog.accept()

    buttons.accepted.connect(accept)
    buttons.rejected.connect(dialog.reject)

    layout.addWidget(QtWidgets.QLabel(
        "Y lines  (column / legend label / rolling mean / transform / color)"))
    layout.addWidget(scroll, 1)
    layout.addWidget(add_button, 0, QtCore.Qt.AlignLeft)
    layout.addWidget(buttons)

    if dialog.exec() != QtWidgets.QDialog.Accepted:
        return None

    def valid(name):
        # Without a loaded file anything typed is kept, so specs can be set up early.
        return bool(name) and (not columns or name in columns)

    lines = [LineSpec(r.column.currentText(), r.label.text() or r.column.currentText(),
                      r.color.property("color"), r.rolling.value(), r.transform.text())
             for r in rows if valid(r.column.currentText())]
    x, z = x_combo.currentText(), z_combo.currentText()
    # replace() keeps any spec field this dialog doesn't show.
    return replace(spec, x=x if valid(x) else None, lines=lines, xlabel=xlabel.text(),
                   ylabel=ylabel.text(), title=title.text(), kind=kind.currentData(),
                   marker_size=marker_size.value(), z=z if valid(z) else None,
                   zlabel=zlabel.text(), legend=legend.isChecked(),
                   show_xlabel=show_x.isChecked(), show_ylabel=show_y.isChecked(),
                   projection=view_3d.currentData(),
                   x_expr=x_transform.text(), z_expr=z_transform.text())


# ---------------------------------------------------------------------------
# Toast (brief warning in the top right)
# ---------------------------------------------------------------------------
def show_toast(parts: AppParts, text: str, cfg: LayoutConfig) -> None:
    for old in parts.window.findChildren(QtWidgets.QLabel, "toast"):
        old.deleteLater()
    toast = QtWidgets.QLabel(text, parts.window)
    toast.setObjectName("toast")
    toast.setStyleSheet(f"background-color: {cfg.toast_color}; color: {cfg.toast_text_color};"
                        "padding: 8px 12px; border-radius: 4px;")
    toast.adjustSize()
    m = cfg.toast_margin_px
    toast.move(parts.window.width() - toast.width() - m, parts.top_bar.height() + m)
    toast.show()
    toast.raise_()
    QtCore.QTimer.singleShot(cfg.toast_ms, toast.deleteLater)


# ---------------------------------------------------------------------------
# Column script window
# ---------------------------------------------------------------------------
class ScriptWindow(QtWidgets.QWidget):
    """Column list (click copies a name) beside a script editor with Save buttons.
    The editor has the usual Ctrl+C/V/X/Z/Y; the main window's Ctrl+Z/Y don't
    reach it, since this is a separate window."""

    def __init__(self, parts: AppParts, cfg: LayoutConfig):
        super().__init__(parts.window, QtCore.Qt.Window)
        self.parts, self.cfg = parts, cfg
        self.setWindowTitle(cfg.script_window_title)
        self.resize(*cfg.script_window_size)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)

        page, self.search, self.columns = build_column_page(cfg)
        self.columns.itemClicked.connect(self.copy_name)
        self.editor = QtWidgets.QPlainTextEdit()
        font = QtGui.QFont(cfg.script_font)
        font.setPointSizeF(cfg.script_font_pt)
        font.setStyleHint(QtGui.QFont.Monospace)
        self.editor.setFont(font)
        self.editor.setPlaceholderText(cfg.script_placeholder)
        self.editor.setLineWrapMode(QtWidgets.QPlainTextEdit.NoWrap)
        self.status = QtWidgets.QLabel()
        self.status.setWordWrap(True)
        self.status.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        save = QtWidgets.QPushButton(cfg.script_save_text)
        save_close = QtWidgets.QPushButton(cfg.script_save_close_text)
        save.clicked.connect(self.save)
        save_close.clicked.connect(self.save_and_close)

        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        split.addWidget(page)
        split.addWidget(self.editor)
        split.setChildrenCollapsible(False)
        w = cfg.script_window_size[0]
        split.setSizes([round(w * cfg.script_columns_frac), round(w * (1 - cfg.script_columns_frac))])
        bottom = QtWidgets.QHBoxLayout()
        bottom.addWidget(self.status, 1)
        bottom.addWidget(save)
        bottom.addWidget(save_close)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(split, 1)
        layout.addLayout(bottom)
        self.refresh()

    def refresh(self) -> None:
        """Re-read the columns, and the script unless there are unsaved edits."""
        fill_column_list(self.columns, self.search, self.parts.state.df)
        if not self.editor.document().isModified():
            self.editor.setPlainText(self.parts.state.script)
            self.editor.document().setModified(False)

    def copy_name(self, item: QtWidgets.QListWidgetItem) -> None:
        QtWidgets.QApplication.clipboard().setText(item.text())
        self.show_status(self.cfg.script_copied_text.format(name=item.text()), error=False)

    def show_status(self, text: str, error: bool) -> None:
        self.status.setStyleSheet(f"color: {self.cfg.script_error_color};" if error else "")
        self.status.setText(text)

    def save(self) -> list[colscript.ScriptError]:
        errors = set_script(self.parts, self.editor.toPlainText(), self.cfg)
        self.editor.document().setModified(False)
        self.refresh()                            # new columns show up in the list
        if errors:
            self.show_status("\n".join(str(e) for e in errors), error=True)
        else:
            count = len(colscript.parse(self.parts.state.script)[0])
            self.show_status(self.cfg.script_ok_text.format(count=count), error=False)
        return errors

    def save_and_close(self) -> None:
        errors = self.save()
        if errors:
            show_toast(self.parts, f"Column script: {len(errors)} line(s) skipped "
                                   f"({errors[0]})", self.cfg)
        self.close()

    def closeEvent(self, event):
        if (self.editor.document().isModified()
                and not confirm(self, self.cfg.script_unsaved_text, self.cfg)):
            event.ignore()
            return
        self.parts.state.script_window = None
        super().closeEvent(event)


def open_script_window(parts: AppParts, cfg: LayoutConfig) -> None:
    window = parts.state.script_window
    if window is None:
        window = parts.state.script_window = ScriptWindow(parts, cfg)
    window.show()
    window.raise_()
    window.activateWindow()


def refresh_script_window(parts: AppParts) -> None:
    if parts.state.script_window is not None:
        parts.state.script_window.refresh()


# ---------------------------------------------------------------------------
# Tabs -- each with its own graph grid
# ---------------------------------------------------------------------------
def new_tab(parts: AppParts, name: str, cfg: LayoutConfig, path: str | None = None) -> Tab:
    """Append a tab with one empty graph (not undoable by itself; callers push_undo)."""
    graphs = build_graph_grid(cfg)
    area = build_graph_area(graphs, lambda: push_undo(parts, cfg), cfg)
    connect_graph_menu(parts, graphs, cfg)
    connect_column_drops(parts, graphs, cfg)
    tab = Tab(name, graphs, area, path)
    tabs = parts.tabs
    tabs.items.append(tab)                        # before insertTab, which may select it
    tabs.stack.addWidget(area)
    tabs.bar.addTab(name)
    set_tab_label(parts, tab)
    return tab


def set_tab_label(parts: AppParts, tab: Tab) -> None:
    i = parts.tabs.items.index(tab)
    parts.tabs.bar.setTabText(i, tab.name)
    parts.tabs.bar.setTabToolTip(i, tab.path or "Not saved to a template file yet")


def drop_tab(parts: AppParts, index: int) -> None:
    """Remove a tab and free its canvas (not undoable by itself)."""
    tab = parts.tabs.items.pop(index)             # before removeTab, which may select another
    parts.tabs.bar.removeTab(index)
    parts.tabs.stack.removeWidget(tab.area)
    tab.graphs.canvas.close()
    tab.area.deleteLater()


def next_tab_name(parts: AppParts, cfg: LayoutConfig) -> str:
    names = {t.name for t in parts.tabs.items}
    n = 1
    while cfg.new_tab_name.format(n=n) in names:
        n += 1
    return cfg.new_tab_name.format(n=n)


def add_tab(parts: AppParts, cfg: LayoutConfig) -> None:
    push_undo(parts, cfg)
    new_tab(parts, next_tab_name(parts, cfg), cfg)
    parts.tabs.bar.setCurrentIndex(len(parts.tabs.items) - 1)


def close_tab(parts: AppParts, index: int, cfg: LayoutConfig) -> None:
    """Tab close button: confirm (if switched on in Settings), then delete. Deleting
    the last tab leaves a fresh empty one."""
    tab = parts.tabs.items[index]
    count = len(all_graphs(tab.graphs))
    if parts.state.confirm_tab_delete and not confirm(
            parts.window, f'Delete tab "{tab.name}" ({count} graph{"s" * (count != 1)})?', cfg):
        return
    push_undo(parts, cfg)
    drop_tab(parts, index)
    if not parts.tabs.items:
        new_tab(parts, next_tab_name(parts, cfg), cfg)


def rename_tab(parts: AppParts, index: int, cfg: LayoutConfig) -> None:
    if index < 0:                                 # double click beside the tabs
        return
    tab = parts.tabs.items[index]
    name, ok = QtWidgets.QInputDialog.getText(parts.window, cfg.window_title,
                                              cfg.rename_tab_text, text=tab.name)
    name = name.strip()
    if ok and name and name != tab.name:
        push_undo(parts, cfg)
        tab.name = name
        set_tab_label(parts, tab)


def fill_tab(parts: AppParts, tab: Tab, snap: TabSnapshot, cfg: LayoutConfig) -> None:
    """Give `tab` the name, template path and graphs of `snap`, and draw it."""
    tab.name, tab.path = snap.name, snap.path
    set_tab_label(parts, tab)
    set_grid_shape(tab.graphs, len(snap.specs), len(snap.specs[0]))
    for row, specs in zip(tab.graphs.cells, snap.specs):
        for graph, spec in zip(row, specs):
            graph.spec = copy.deepcopy(spec)
    tab.graphs.links = links_from_positions(tab.graphs, snap.links)
    redraw_grid(parts, tab.graphs, cfg)


def set_tabs(parts: AppParts, snaps: list[TabSnapshot], cfg: LayoutConfig) -> None:
    """Make the open tabs match `snaps` (reusing existing tabs by position)."""
    while len(parts.tabs.items) > len(snaps):
        drop_tab(parts, len(parts.tabs.items) - 1)
    while len(parts.tabs.items) < len(snaps):
        new_tab(parts, "", cfg)
    for tab, snap in zip(parts.tabs.items, snaps):
        fill_tab(parts, tab, snap, cfg)


# ---------------------------------------------------------------------------
# Templates -- one JSON file per tab
# ---------------------------------------------------------------------------
def templates_dir(cfg: LayoutConfig) -> str:
    """The default template folder (created if missing)."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), cfg.templates_dir)
    os.makedirs(path, exist_ok=True)
    return path


def template_filter(cfg: LayoutConfig) -> str:
    return f"Tab templates (*{cfg.template_suffix})"


def safe_file_name(name: str) -> str:
    return re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name).strip(" .") or "tab"


def write_template(tab: Tab, path: str, cfg: LayoutConfig) -> None:
    data = {"format": cfg.template_format, "version": 1, "name": tab.name,
            "graphs": [[asdict(g.spec) for g in row] for row in tab.graphs.cells],
            "links": [{"axis": axis, "graphs": [list(cell) for cell in cells]}
                      for axis, cells in link_positions(tab.graphs)]}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _known(cls, data: dict) -> dict:
    """The keys of `data` that are fields of dataclass `cls` (older/newer files
    may have fewer or extra keys)."""
    names = {f.name for f in fields(cls)}
    return {k: v for k, v in data.items() if k in names}


def _legacy_expr(d: dict, prefix: str) -> str:
    """Transform text for files saved with the earlier scale/offset/divide fields."""
    scale, offset = d.get(prefix + "scale", 1), d.get(prefix + "offset", 0)
    parts = []
    if scale != 1 and not (scale == 0 and d.get(prefix + "divide")):
        parts.append(f"{'/' if d.get(prefix + 'divide') else '*'} {scale:g}")
    if offset:
        parts.append(f"{'-' if offset < 0 else '+'} {abs(offset):g}")
    return " ".join(parts)


def read_template(path: str, cfg: LayoutConfig) -> TabSnapshot:
    """Raises ValueError if the file isn't a tab template."""
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict) or data.get("format") != cfg.template_format:
            raise ValueError("not a tab template")
        grid = data["graphs"]
        if not grid or not grid[0] or any(len(row) != len(grid[0]) for row in grid):
            raise ValueError("graph grid is empty or not rectangular")
        specs = [[GraphSpec(**{**_known(GraphSpec, d),
                               "x_expr": d.get("x_expr") or _legacy_expr(d, "x_"),
                               "z_expr": d.get("z_expr") or _legacy_expr(d, "z_"),
                               "lines": [LineSpec(**{**_known(LineSpec, line),
                                                     "expr": line.get("expr")
                                                     or _legacy_expr(line, "")})
                                         for line in d.get("lines", [])]})
                  for d in row] for row in grid]
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as e:
        raise ValueError(str(e) or type(e).__name__) from e
    name = data.get("name") or os.path.splitext(os.path.basename(path))[0]
    try:
        links = tuple((str(link["axis"]), tuple((int(r), int(c)) for r, c in link["graphs"]))
                      for link in data.get("links", []))
    except (KeyError, TypeError, ValueError):
        links = ()                                # a bad links entry: keep the graphs
    return TabSnapshot(str(name), path, specs, links)


def template_files(folder: str, cfg: LayoutConfig) -> list[str]:
    return sorted(os.path.join(folder, f) for f in os.listdir(folder)
                  if f.lower().endswith(cfg.template_suffix))


def load_templates(parts: AppParts, paths: list[str], replace: bool,
                   cfg: LayoutConfig) -> None:
    """Open each template file as a tab, after the open tabs or (replace) instead of
    them. One Ctrl+Z step. Files that aren't templates are skipped with a toast."""
    snaps, skipped = [], []
    for path in paths:
        try:
            snaps.append(read_template(path, cfg))
        except ValueError as e:
            skipped.append(f"{os.path.basename(path)} ({e})")
    if snaps:
        push_undo(parts, cfg)
        if replace:
            set_tabs(parts, snaps, cfg)
            first = 0
        else:
            first = len(parts.tabs.items)
            for snap in snaps:
                fill_tab(parts, new_tab(parts, snap.name, cfg), snap, cfg)
        parts.tabs.bar.setCurrentIndex(first)
    if skipped:
        show_toast(parts, f"Skipped {len(skipped)} file(s): " + "; ".join(skipped), cfg)
    elif not snaps:
        show_toast(parts, "No tab templates found", cfg)


def choose_templates(parts: AppParts, replace: bool, cfg: LayoutConfig) -> None:
    paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
        parts.window, cfg.load_template_text, templates_dir(cfg), template_filter(cfg))
    if paths:
        load_templates(parts, paths, replace, cfg)


def choose_template_folder(parts: AppParts, replace: bool, cfg: LayoutConfig) -> None:
    folder = QtWidgets.QFileDialog.getExistingDirectory(
        parts.window, cfg.load_template_folder_text, templates_dir(cfg))
    if folder:
        load_templates(parts, template_files(folder, cfg), replace, cfg)


def save_tab(parts: AppParts, tab: Tab, path: str, cfg: LayoutConfig) -> bool:
    try:
        write_template(tab, path, cfg)
    except OSError as e:
        show_toast(parts, f"Could not save {os.path.basename(path)}: {e}", cfg)
        return False
    tab.path = path
    set_tab_label(parts, tab)
    return True


def save_template(parts: AppParts, cfg: LayoutConfig) -> None:
    """Save the current tab to its template file, or ask where if it has none."""
    tab = parts.tabs.current()
    if tab.path is None:
        save_template_as(parts, cfg)
    elif save_tab(parts, tab, tab.path, cfg):
        show_toast(parts, f"Saved {os.path.basename(tab.path)}", cfg)


def save_template_as(parts: AppParts, cfg: LayoutConfig) -> None:
    tab = parts.tabs.current()
    start = tab.path or os.path.join(templates_dir(cfg), safe_file_name(tab.name)
                                     + cfg.template_suffix)
    path, _ = QtWidgets.QFileDialog.getSaveFileName(parts.window, cfg.save_template_as_text,
                                                    start, template_filter(cfg))
    if not path:
        return
    if not path.lower().endswith(cfg.template_suffix):
        path += cfg.template_suffix
    if save_tab(parts, tab, path, cfg):
        show_toast(parts, f"Saved {os.path.basename(path)}", cfg)


def save_all_templates(parts: AppParts, cfg: LayoutConfig) -> None:
    """Save every tab into one folder, a file per tab named after the tab."""
    folder = QtWidgets.QFileDialog.getExistingDirectory(
        parts.window, cfg.save_all_templates_text, templates_dir(cfg))
    if not folder:
        return
    paths, used = [], set()
    for tab in parts.tabs.items:
        base = safe_file_name(tab.name)
        name, n = base, 2
        while name.lower() in used:               # two tabs with the same name
            name, n = f"{base} ({n})", n + 1
        used.add(name.lower())
        paths.append(os.path.join(folder, name + cfg.template_suffix))
    existing = [p for p in paths if os.path.exists(p)]
    if existing and not confirm(parts.window, f"Overwrite {len(existing)} existing "
                                              f"template file(s) in {folder}?", cfg):
        return
    saved = sum(save_tab(parts, tab, path, cfg) for tab, path in zip(parts.tabs.items, paths))
    show_toast(parts, f"Saved {saved} template(s) to {folder}", cfg)


# ---------------------------------------------------------------------------
# Axis-link editing overlay
# ---------------------------------------------------------------------------
@dataclass
class LinkEdit:
    """An open links overlay: working copies of the tab's links (applied on Done)."""
    graphs: GraphGrid
    links: list[AxisLink]
    overlay: "LinkOverlay"
    escape: QtGui.QShortcut
    page_before: int                              # side-bar page to go back to
    sizes_before: list[int]                       # splitter sizes to go back to
    redraw: object                                # canvas draw handler (to disconnect)


class LinkOverlay(QtWidgets.QWidget):
    """Covers the graph canvas while links are edited: draws a translucent box on
    each graph of the selected link and turns clicks into checking/unchecking.
    It also keeps clicks from zooming the graphs underneath."""

    def __init__(self, parts: AppParts, graphs: GraphGrid, cfg: LayoutConfig):
        native = graphs.canvas.native
        super().__init__(native)
        self.parts, self.graphs, self.cfg = parts, graphs, cfg
        self.setGeometry(native.rect())
        self.setCursor(QtCore.Qt.PointingHandCursor)
        native.installEventFilter(self)           # follow the canvas's size
        self.show()

    def eventFilter(self, obj, event):
        if event.type() == QtCore.QEvent.Resize:
            self.setGeometry(obj.rect())
        return False

    def selected(self) -> AxisLink | None:
        edit, row = self.parts.state.link_edit, self.parts.side.links.list.currentRow()
        return edit.links[row] if edit is not None and 0 <= row < len(edit.links) else None

    def paintEvent(self, event):
        cfg, link = self.cfg, self.selected()
        painter = QtGui.QPainter(self)
        gap = cfg.link_box_gap_px / 2
        for graph in all_graphs(self.graphs):
            x, y, w, h = canvas_rect(graph.cell)
            box = QtCore.QRectF(x + gap, y + gap, w - 2 * gap, h - 2 * gap)
            if not linkable(graph):
                painter.fillRect(box, QtGui.QColor(*cfg.link_3d_color))
            elif link is not None and graph in link.graphs:
                painter.fillRect(box, QtGui.QColor(*cfg.link_box_color))
                painter.setPen(QtGui.QPen(QtGui.QColor(*cfg.link_box_border), 2))
                painter.drawRect(box)
                painter.drawText(box.adjusted(8, 6, 0, 0), link_name(self.parts, link))
            else:
                painter.setPen(QtGui.QPen(QtGui.QColor(*cfg.link_free_border), 1,
                                          QtCore.Qt.DashLine))
                painter.drawRect(box)
        painter.end()

    def mousePressEvent(self, event):
        if event.button() != QtCore.Qt.LeftButton:
            return
        pos = event.position()
        graph = graph_at(self.graphs, (pos.x(), pos.y()))
        link = self.selected()
        if graph is None:
            return
        if not linkable(graph):
            show_toast(self.parts, "3D graphs can't be linked", self.cfg)
        elif link is None:
            show_toast(self.parts, "Add a link (or select one) first", self.cfg)
        else:
            if graph in link.graphs:
                link.graphs.remove(graph)
            else:
                link.graphs.append(graph)
            refresh_link_list(self.parts, self.cfg)
        self.update()


def link_name(parts: AppParts, link: AxisLink) -> str:
    edit = parts.state.link_edit
    same_axis = [l for l in edit.links if l.axis == link.axis] if edit else [link]
    return f"{link.axis} link {same_axis.index(link) + 1}"


def refresh_link_list(parts: AppParts, cfg: LayoutConfig) -> None:
    """Rebuild the side-bar list from the working links, keeping the selection."""
    edit, widget = parts.state.link_edit, parts.side.links.list
    row = widget.currentRow()
    widget.blockSignals(True)
    widget.clear()
    for link in edit.links:
        item = QtWidgets.QListWidgetItem()
        widget.addItem(item)
        line = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(line)
        layout.setContentsMargins(6, 2, 2, 2)
        n = len(link.graphs)
        layout.addWidget(QtWidgets.QLabel(f"{link_name(parts, link)}  "
                                          f"({n} graph{'s' * (n != 1)})"), 1)
        remove = QtWidgets.QToolButton()
        remove.setText("\u2715")
        remove.setToolTip("Delete this link")
        remove.setAutoRaise(True)
        remove.clicked.connect(lambda checked=False, l=link: delete_link(parts, l, cfg))
        layout.addWidget(remove)
        item.setSizeHint(line.sizeHint())
        widget.setItemWidget(item, line)
    widget.setCurrentRow(min(row, len(edit.links) - 1))
    widget.blockSignals(False)
    edit.overlay.update()


def add_link(parts: AppParts, axis: str, cfg: LayoutConfig) -> None:
    edit = parts.state.link_edit
    edit.links.append(AxisLink(axis))
    refresh_link_list(parts, cfg)
    parts.side.links.list.setCurrentRow(len(edit.links) - 1)   # new link is selected


def delete_link(parts: AppParts, link: AxisLink, cfg: LayoutConfig) -> None:
    parts.state.link_edit.links.remove(link)
    refresh_link_list(parts, cfg)


def start_link_edit(parts: AppParts, cfg: LayoutConfig) -> None:
    """Open the overlay on the current tab: side bar shows its links."""
    if parts.state.link_edit is not None:
        return
    graphs, side = parts.graphs, parts.side
    overlay = LinkOverlay(parts, graphs, cfg)
    escape = QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Escape), parts.window)
    escape.activated.connect(lambda: finish_link_edit(parts, False, cfg))
    redraw = lambda event: overlay.update()      # graphs moved (e.g. a row added)
    graphs.canvas.events.draw.connect(redraw)
    parts.state.link_edit = LinkEdit(graphs, [AxisLink(l.axis, list(l.graphs))
                                              for l in graphs.links],
                                     overlay, escape, side.stack.currentIndex(),
                                     parts.splitter.sizes(), redraw)
    sizes = parts.splitter.sizes()
    if sizes[0] < cfg.links_side_bar_px:          # room for the links page
        wide = min(cfg.links_side_bar_px, side.frame.maximumWidth())
        parts.splitter.setSizes([wide, sum(sizes) - wide])
    for widget in (side.files_button, side.columns_button):
        widget.hide()
    side.stack.setCurrentIndex(LINKS_PAGE)
    side.links.done.setChecked(False)
    for widget in (parts.tabs.bar.parentWidget(), parts.file_button, parts.links_button):
        widget.setEnabled(False)                  # no tab/file changes under the overlay
    refresh_link_list(parts, cfg)
    side.links.list.setCurrentRow(0 if parts.state.link_edit.links else -1)


def finish_link_edit(parts: AppParts, apply: bool, cfg: LayoutConfig) -> None:
    """Close the overlay. apply: merge overlapping links and keep them (one undo
    step, and linked axes are fitted to their data); otherwise discard the edits."""
    edit = parts.state.link_edit
    if edit is None:
        return
    graphs = edit.graphs
    parts.state.link_edit = None
    if apply:
        links = merged_links(graphs, edit.links)
        if [(l.axis, l.graphs) for l in links] != [(l.axis, l.graphs) for l in graphs.links]:
            push_undo(parts, cfg)
            graphs.links = links
            fit_links(graphs)
            graphs.canvas.update()
    graphs.canvas.events.draw.disconnect(edit.redraw)
    edit.overlay.deleteLater()
    edit.escape.setEnabled(False)
    edit.escape.deleteLater()
    side = parts.side
    side.links.list.clear()
    for widget in (side.files_button, side.columns_button):
        widget.show()
    side.stack.setCurrentIndex(edit.page_before)
    parts.splitter.setSizes(edit.sizes_before)
    for widget in (parts.tabs.bar.parentWidget(), parts.file_button, parts.links_button):
        widget.setEnabled(True)


def connect_links_page(parts: AppParts, cfg: LayoutConfig) -> None:
    links = parts.side.links
    links.add_x.clicked.connect(lambda: add_link(parts, "x", cfg))
    links.add_y.clicked.connect(lambda: add_link(parts, "y", cfg))
    links.done.toggled.connect(lambda on: finish_link_edit(parts, True, cfg) if on else None)
    links.cancel.clicked.connect(lambda: finish_link_edit(parts, False, cfg))
    links.list.currentRowChanged.connect(
        lambda row: parts.state.link_edit and parts.state.link_edit.overlay.update())
    parts.links_button.clicked.connect(lambda: start_link_edit(parts, cfg))


# ---------------------------------------------------------------------------
# Folder selection
# ---------------------------------------------------------------------------
def apply_data_dir(parts: AppParts, path: str, cfg: LayoutConfig) -> None:
    """Point the app at `path`: tooltip, file tree, and a warning if misnamed."""
    parts.file_button.setToolTip(f"Data folder:\n{path}")
    set_tree_root(parts.side.tree, path, cfg)
    if not is_data_folder(path, cfg):
        show_toast(parts, f'Selected folder is not named "{cfg.data_folder_name}"', cfg)


def choose_data_dir(parts: AppParts, cfg: LayoutConfig) -> None:
    start = load_data_dir(cfg) or os.path.expanduser("~")
    path = QtWidgets.QFileDialog.getExistingDirectory(parts.window, cfg.select_button_text, start)
    if path:
        save_data_dir(path, cfg)
        apply_data_dir(parts, path, cfg)


# ---------------------------------------------------------------------------
# Builders -- each takes the config and returns a widget, no hidden state
# ---------------------------------------------------------------------------
def make_panel(color: str, cfg: LayoutConfig) -> QtWidgets.QFrame:
    panel = QtWidgets.QFrame()
    panel.setStyleSheet(f"background-color: {color}; color: {cfg.text_color};")
    return panel


def build_top_bar(cfg: LayoutConfig) -> QtWidgets.QFrame:
    bar = make_panel(cfg.top_bar_color, cfg)
    # Ignored vertical policy so the bar follows its stretch share exactly
    # instead of being held taller by its contents' size hints.
    bar.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Ignored)
    layout = QtWidgets.QHBoxLayout(bar)
    layout.setContentsMargins(8, 2, 8, 2)
    return bar


def build_bar_button(text: str) -> QtWidgets.QPushButton:
    button = QtWidgets.QPushButton(text)
    button.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Ignored)
    return button


def build_file_menu(parts: AppParts, cfg: LayoutConfig) -> QtWidgets.QMenu:
    """Template loading/saving, with the data folder choice at the bottom."""
    menu = QtWidgets.QMenu(parts.file_button)
    menu.addAction(cfg.load_template_text, lambda: choose_templates(parts, False, cfg))
    menu.addAction(cfg.load_template_folder_text,
                   lambda: choose_template_folder(parts, False, cfg))
    menu.addAction(cfg.replace_template_text, lambda: choose_templates(parts, True, cfg))
    menu.addAction(cfg.replace_template_folder_text,
                   lambda: choose_template_folder(parts, True, cfg))
    menu.addSeparator()
    menu.addAction(cfg.save_template_text, lambda: save_template(parts, cfg))
    menu.addAction(cfg.save_template_as_text, lambda: save_template_as(parts, cfg))
    menu.addAction(cfg.save_all_templates_text, lambda: save_all_templates(parts, cfg))
    menu.addSeparator()
    menu.addAction(cfg.select_button_text, lambda: choose_data_dir(parts, cfg))
    return menu


def build_switch_action(menu: QtWidgets.QMenu, text: str, on: bool, on_toggle,
                        cfg: LayoutConfig) -> "ToggleSwitchAction":
    """Menu row "text  [switch]"; clicking it flips the switch without closing the menu."""
    action = ToggleSwitchAction(menu, text, on, cfg)
    action.switch.toggled.connect(on_toggle)
    menu.addAction(action)
    return action


def build_settings_menu(parts: AppParts, cfg: LayoutConfig) -> QtWidgets.QMenu:
    state, menu = parts.state, QtWidgets.QMenu(parts.top_bar)   # parented, so it isn't GC'd

    def setter(attr, key):
        def toggle(on):
            setattr(state, attr, on)
            _settings(cfg).setValue(key, on)
        return toggle

    build_switch_action(menu, cfg.advanced_text, state.advanced,
                        lambda on: set_advanced(parts, on, cfg), cfg)
    build_switch_action(menu, cfg.confirm_tab_delete_text, state.confirm_tab_delete,
                        setter("confirm_tab_delete", cfg.settings_confirm_tab_key), cfg)
    build_switch_action(menu, cfg.confirm_graph_delete_text, state.confirm_graph_delete,
                        setter("confirm_graph_delete", cfg.settings_confirm_graph_key), cfg)
    return menu


class TabBar(QtWidgets.QTabBar):
    """QTabBar whose minimum width never exceeds its tabs' width. Qt's minimum
    reserves room for the scroll arrows, which with one short tab left a gap
    before the "+" button."""

    def minimumSizeHint(self):
        hint = super().minimumSizeHint()
        hint.setWidth(min(hint.width(), self.sizeHint().width()))
        return hint


def build_tab_strip(cfg: LayoutConfig) -> tuple[QtWidgets.QWidget, QtWidgets.QTabBar,
                                                QtWidgets.QToolButton]:
    """Tab bar with a "+" button after the tabs."""
    strip = make_panel(cfg.top_bar_color, cfg)
    strip.setFixedHeight(cfg.tab_bar_px)
    bar = TabBar()
    bar.setTabsClosable(True)
    bar.setExpanding(False)
    bar.setDocumentMode(True)
    bar.setDrawBase(False)
    bar.setStyleSheet(
        f"QTabBar::tab {{ background: {cfg.tab_color}; color: {cfg.text_color};"
        " padding: 4px 10px; border: none; }"
        f"QTabBar::tab:selected {{ background: {cfg.tab_checked_color}; }}")
    add = QtWidgets.QToolButton()
    add.setText("+")
    add.setToolTip(cfg.add_tab_tooltip)
    add.setStyleSheet(f"QToolButton {{ border: none; padding: 0 8px; color: {cfg.text_color}; }}"
                      f"QToolButton:hover {{ background: {cfg.plus_hover_color}; }}")
    layout = QtWidgets.QHBoxLayout(strip)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(0)
    layout.addWidget(bar)
    layout.addWidget(add)
    layout.addStretch(1)
    return strip, bar, add


def build_file_label(cfg: LayoutConfig) -> QtWidgets.QLabel:
    label = QtWidgets.QLabel("No file loaded")
    label.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Ignored)
    return label


def build_tab_button(text: str, cfg: LayoutConfig) -> QtWidgets.QPushButton:
    button = QtWidgets.QPushButton(text)
    button.setCheckable(True)
    button.setFixedHeight(cfg.side_tab_px)
    button.setStyleSheet(
        f"QPushButton {{ background-color: {cfg.tab_color}; border: none; }}"
        f"QPushButton:checked {{ background-color: {cfg.tab_checked_color}; }}")
    return button


def build_side_bar(cfg: LayoutConfig) -> SideBar:
    frame = make_panel(cfg.side_bar_color, cfg)
    frame.setMinimumWidth(cfg.side_bar_min_px)

    files_button = build_tab_button(cfg.files_tab_text, cfg)
    columns_button = build_tab_button(cfg.columns_tab_text, cfg)
    tree = build_file_tree(cfg)
    refresh = QtWidgets.QToolButton()
    refresh.setText(cfg.tree_refresh_text)
    refresh.setToolTip(cfg.tree_refresh_tooltip)
    refresh.setAutoRaise(True)
    refresh.clicked.connect(lambda: refresh_tree(tree, cfg))
    files_page = QtWidgets.QWidget()
    files_layout = QtWidgets.QVBoxLayout(files_page)
    files_layout.setContentsMargins(0, 0, 0, 0)
    files_layout.setSpacing(0)
    files_layout.addWidget(refresh, 0, QtCore.Qt.AlignRight)
    files_layout.addWidget(tree, 1)
    column_page, column_search, column_list = build_column_page(cfg)

    links_page, links = build_links_page(cfg)

    stack = QtWidgets.QStackedWidget()
    stack.insertWidget(FILES_PAGE, files_page)
    stack.insertWidget(COLUMNS_PAGE, column_page)
    stack.insertWidget(LINKS_PAGE, links_page)

    # Exclusive group: exactly one tab button is checked, and it picks the page.
    tabs = QtWidgets.QButtonGroup(frame)
    tabs.addButton(files_button, FILES_PAGE)
    tabs.addButton(columns_button, COLUMNS_PAGE)
    tabs.idClicked.connect(stack.setCurrentIndex)
    files_button.setChecked(True)

    tab_row = QtWidgets.QHBoxLayout()
    tab_row.setSpacing(0)
    tab_row.addWidget(files_button)
    tab_row.addWidget(columns_button)

    layout = QtWidgets.QVBoxLayout(frame)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(0)
    layout.addLayout(tab_row)
    layout.addWidget(stack)
    return SideBar(frame, files_button, columns_button, stack, tree, column_search, column_list,
                   links)


def build_links_page(cfg: LayoutConfig) -> tuple[QtWidgets.QWidget, LinksPage]:
    page = QtWidgets.QWidget()
    done = QtWidgets.QCheckBox(cfg.links_done_text)
    cancel = QtWidgets.QPushButton(cfg.links_cancel_text)
    add_x = QtWidgets.QPushButton(cfg.add_x_link_text)
    add_y = QtWidgets.QPushButton(cfg.add_y_link_text)
    help_text = QtWidgets.QLabel(cfg.links_help_text)
    help_text.setWordWrap(True)
    links = QtWidgets.QListWidget()
    links.setFrameShape(QtWidgets.QFrame.NoFrame)

    top = QtWidgets.QHBoxLayout()
    top.addWidget(done)
    top.addStretch(1)
    top.addWidget(cancel)
    adds = QtWidgets.QHBoxLayout()
    adds.addWidget(add_x)
    adds.addWidget(add_y)
    layout = QtWidgets.QVBoxLayout(page)
    layout.setContentsMargins(4, 4, 4, 4)
    layout.addLayout(top)
    layout.addLayout(adds)
    layout.addWidget(help_text)
    layout.addWidget(links, 1)
    return page, LinksPage(done, cancel, add_x, add_y, links)


def build_splitter(side_bar: QtWidgets.QWidget, main: QtWidgets.QWidget,
                   cfg: LayoutConfig) -> QtWidgets.QSplitter:
    splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
    splitter.addWidget(side_bar)
    splitter.addWidget(main)
    splitter.setChildrenCollapsible(False)
    splitter.setOpaqueResize(cfg.live_side_bar_resize)
    splitter.setHandleWidth(cfg.splitter_handle_px)
    splitter.setStyleSheet(f"QSplitter::handle {{ background-color: {cfg.handle_color}; }}")
    splitter.setStretchFactor(0, 1 if cfg.side_bar_grows_on_resize else 0)
    splitter.setStretchFactor(1, 1)
    # Before the window is shown, setSizes is treated as relative weights.
    splitter.setSizes([round(cfg.side_bar_frac * 1000), round((1 - cfg.side_bar_frac) * 1000)])
    return splitter


def clamp_side_bar(splitter: QtWidgets.QSplitter, side_bar: QtWidgets.QWidget,
                   cfg: LayoutConfig) -> None:
    """Keep the side bar under side_bar_max_frac of the splitter width."""
    side_bar.setMaximumWidth(max(cfg.side_bar_min_px,
                                 int(splitter.width() * cfg.side_bar_max_frac)))


class ToggleSwitch(QtWidgets.QAbstractButton):
    """Small on/off pill switch (a checkable button)."""

    def __init__(self, cfg: LayoutConfig):
        super().__init__()
        self.cfg = cfg
        self.setCheckable(True)
        self.setCursor(QtCore.Qt.PointingHandCursor)
        self.setFixedSize(*cfg.switch_size_px)

    def paintEvent(self, event):
        w, h = self.width(), self.height()
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        p.setPen(QtCore.Qt.NoPen)
        on = self.isChecked()
        p.setBrush(QtGui.QColor(self.cfg.switch_on_color if on else self.cfg.switch_off_color))
        p.drawRoundedRect(0, 0, w, h, h / 2, h / 2)
        knob, m = h - 4, 2
        p.setBrush(QtGui.QColor(self.cfg.switch_knob_color))
        p.drawEllipse(w - knob - m if on else m, m, knob, knob)
        p.end()


class ToggleSwitchAction(QtWidgets.QWidgetAction):
    """A menu row with a label and a ToggleSwitch; a click anywhere on the row flips it."""

    def __init__(self, menu: QtWidgets.QMenu, text: str, on: bool, cfg: LayoutConfig):
        super().__init__(menu)
        row = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(row)
        layout.setContentsMargins(10, 4, 10, 4)
        label = QtWidgets.QLabel(text)
        self.switch = ToggleSwitch(cfg)
        self.switch.setChecked(on)
        layout.addWidget(label, 1)
        layout.addSpacing(16)
        layout.addWidget(self.switch)
        row.mouseReleaseEvent = lambda event: self.switch.toggle()
        self.setDefaultWidget(row)


def load_flag(key: str, default: bool, cfg: LayoutConfig) -> bool:
    return _settings(cfg).value(key, default, type=bool)


def load_advanced(cfg: LayoutConfig) -> bool:
    return load_flag(cfg.settings_advanced_key, False, cfg)


def load_script(cfg: LayoutConfig) -> str:
    return _settings(cfg).value(cfg.settings_script_key, "", type=str)


def set_advanced(parts: AppParts, on: bool, cfg: LayoutConfig) -> None:
    """Advanced-mode switch: remembered between runs. Menus and dialogs check it when
    they open; the script button is shown/hidden here."""
    parts.state.advanced = on
    _settings(cfg).setValue(cfg.settings_advanced_key, on)
    parts.script_button.setVisible(feature_on(parts.state, "script", cfg))
    parts.links_button.setVisible(feature_on(parts.state, "links", cfg))


def build_window(cfg: LayoutConfig = LAYOUT) -> AppParts:
    window = QtWidgets.QMainWindow()
    window.setWindowTitle(cfg.window_title)
    window.resize(*cfg.window_size)

    top_bar = build_top_bar(cfg)
    file_button = build_bar_button(cfg.file_menu_text)
    file_button.setToolTip("No data folder selected")
    file_label = build_file_label(cfg)
    top_bar.layout().addWidget(file_button)
    top_bar.layout().addSpacing(12)
    top_bar.layout().addWidget(file_label)
    top_bar.layout().addStretch(1)
    links_button = build_bar_button(cfg.links_button_text)
    script_button = build_bar_button(cfg.script_button_text)
    settings_button = build_bar_button(cfg.settings_text)
    top_bar.layout().addWidget(links_button)
    top_bar.layout().addSpacing(12)
    top_bar.layout().addWidget(script_button)
    top_bar.layout().addSpacing(12)
    top_bar.layout().addWidget(settings_button)

    side = build_side_bar(cfg)
    tab_strip, tab_bar, add_tab_button = build_tab_strip(cfg)
    tabs = Tabs(tab_bar, QtWidgets.QStackedWidget())
    main_area = QtWidgets.QWidget()
    main_layout = QtWidgets.QVBoxLayout(main_area)
    main_layout.setContentsMargins(0, 0, 0, 0)
    main_layout.setSpacing(0)
    main_layout.addWidget(tab_strip)
    main_layout.addWidget(tabs.stack, 1)
    splitter = build_splitter(side.frame, main_area, cfg)

    root = QtWidgets.QWidget()
    layout = QtWidgets.QVBoxLayout(root)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(0)
    top_pct = round(cfg.top_bar_frac * 100)
    layout.addWidget(top_bar, stretch=top_pct)
    layout.addWidget(splitter, stretch=100 - top_pct)
    window.setCentralWidget(root)

    # Re-clamp the side bar's max width whenever the splitter is resized.
    splitter.resizeEvent = _chain_resize(splitter.resizeEvent,
                                         lambda: clamp_side_bar(splitter, side.frame, cfg))

    state = AppState(
        script=load_script(cfg),
        confirm_tab_delete=load_flag(cfg.settings_confirm_tab_key,
                                     cfg.confirm_tab_delete_default, cfg),
        confirm_graph_delete=load_flag(cfg.settings_confirm_graph_key,
                                       cfg.require_confirm_on_graph_delete, cfg))
    parts = AppParts(window, top_bar, side, splitter, tabs, file_button, file_label,
                     script_button, links_button, state)
    connect_links_page(parts, cfg)
    set_advanced(parts, load_advanced(cfg), cfg)
    file_button.setMenu(build_file_menu(parts, cfg))
    settings_button.setMenu(build_settings_menu(parts, cfg))
    script_button.clicked.connect(lambda: open_script_window(parts, cfg))
    side.tree.clicked.connect(lambda index: on_tree_clicked(side.tree, index))
    side.tree.doubleClicked.connect(lambda index: on_tree_double_clicked(parts, index, cfg))

    tab_bar.currentChanged.connect(
        lambda i: tabs.stack.setCurrentWidget(tabs.items[i].area) if i >= 0 else None)
    tab_bar.tabCloseRequested.connect(lambda i: close_tab(parts, i, cfg))
    tab_bar.tabBarDoubleClicked.connect(lambda i: rename_tab(parts, i, cfg))
    add_tab_button.clicked.connect(lambda: add_tab(parts, cfg))
    new_tab(parts, next_tab_name(parts, cfg), cfg)

    # Window-wide; a focused text box still gets Ctrl+Z / Ctrl+Y for its own text first.
    for keys, action in (("Ctrl+Z", lambda: step_history(parts, True, cfg)),
                         ("Ctrl+Y", lambda: step_history(parts, False, cfg)),
                         ("Ctrl+S", lambda: save_template(parts, cfg))):
        QtGui.QShortcut(QtGui.QKeySequence(keys), window).activated.connect(action)
    return parts


def _chain_resize(original, callback):
    def handler(event):
        original(event)
        callback()
    return handler


# ---------------------------------------------------------------------------
def main() -> int:
    qt_app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    parts = build_window(LAYOUT)
    parts.window.show()
    saved = load_data_dir(LAYOUT)
    if saved:
        # Deferred so the toast (if any) is placed against the shown window size.
        QtCore.QTimer.singleShot(0, lambda: apply_data_dir(parts, saved, LAYOUT))
    return qt_app.exec()


if __name__ == "__main__":
    sys.exit(main())
