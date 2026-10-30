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
import os
import sys
from dataclasses import dataclass, field
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

    select_button_text: str = "Select data folder"
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
    graph_spacing_px: int = 6                     # gap between neighbouring graphs
    plus_strip_px: int = 18                       # thickness of the "+" strips

    # Graph text, in points (pixel sizes follow from vplot.SCREEN_DPI).
    title_fontsize: float = 14.4
    label_fontsize: float = 12.0
    tick_fontsize: float = 10.2
    legend_fontsize: float = 10.2
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
    default_graph_kind: str = "scatter"           # type of a new graph
    default_marker_size: float = 1.0              # scatter point diameter, points
    marker_size_range: tuple[float, float] = (0.5, 30.0)

    # Advanced mode (switch in the top right). With it off, these features are hidden:
    #   graph_type    graph type / point size / z column in the Configure dialog
    #   copy_graph    "Copy graph to" in the graph menu
    #   copy_y        "Copy y columns to" in the graph menu
    #   custom_color  "Custom color..." (the 20 preset colors stay)
    #   script        the "Script" button (new columns from formulas, see FSLib/colscript.py)
    advanced_features: tuple[str, ...] = ("graph_type", "copy_graph", "copy_y", "custom_color",
                                          "script")
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
    graph_dialog_size: tuple[int, int] = (640, 480)
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


def is_3d(kind: str) -> bool:
    return kind == "scatter3d"


@dataclass(eq=False)                              # compare graphs by identity
class Graph:
    ax: "vp.Axes | vp.Axes3D | None"
    cell: scene.widgets.Grid                      # the grid cell the axes lives in
    spec: GraphSpec = field(default_factory=GraphSpec)


@dataclass
class GraphGrid:
    """One SceneCanvas holding a rows x cols grid of graphs."""
    canvas: scene.SceneCanvas
    grid: scene.widgets.Grid
    cells: list[list[Graph]] = field(default_factory=list)
    zoom_history: list[Graph] = field(default_factory=list)   # graph of each zoom, oldest first
    rubber_band: QtWidgets.QRubberBand | None = None          # box-zoom outline over the canvas


class Snapshot(NamedTuple):
    """What Ctrl+Z restores: the loaded file, the column script and every graph's
    spec, by grid position."""
    df_path: str | None
    script: str
    specs: list[list[GraphSpec]]


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
    script_window: QtWidgets.QWidget | None = None       # open ScriptWindow, if any


def feature_on(state: AppState, name: str, cfg: LayoutConfig) -> bool:
    """False for an advanced-only feature (cfg.advanced_features) while advanced mode is off."""
    return state.advanced or name not in cfg.advanced_features


class SideBar(NamedTuple):
    """Left bar: Files/Columns tab buttons over a stack of the two pages."""
    frame: QtWidgets.QFrame
    files_button: QtWidgets.QPushButton
    columns_button: QtWidgets.QPushButton
    stack: QtWidgets.QStackedWidget
    tree: QtWidgets.QTreeView
    column_search: QtWidgets.QLineEdit
    column_list: QtWidgets.QListWidget


FILES_PAGE, COLUMNS_PAGE = 0, 1


class AppParts(NamedTuple):
    """Handles to the pieces of the window, so other code can fill them in."""
    window: QtWidgets.QMainWindow
    top_bar: QtWidgets.QFrame
    side: SideBar
    splitter: QtWidgets.QSplitter
    graphs: GraphGrid
    select_button: QtWidgets.QPushButton
    file_label: QtWidgets.QLabel
    script_button: QtWidgets.QPushButton
    state: AppState


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
def snapshot(parts: AppParts) -> Snapshot:
    return Snapshot(parts.state.df_path, parts.state.script,
                    [[copy.deepcopy(g.spec) for g in row] for row in parts.graphs.cells])


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
    graphs = parts.graphs
    set_grid_shape(graphs, len(snap.specs), len(snap.specs[0]))
    for row, specs in zip(graphs.cells, snap.specs):
        for graph, spec in zip(row, specs):
            graph.spec = copy.deepcopy(spec)
    redraw_all(parts, cfg)
    return True


def step_history(parts: AppParts, back: bool, cfg: LayoutConfig) -> None:
    """Ctrl+Z (back=True) / Ctrl+Y: move one snapshot between the undo and redo stacks."""
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
    """Hides children of `root_path` that aren't folders starting with `prefix`.
    Set `prefix` to None to show everything."""

    def __init__(self):
        super().__init__()
        self.root_path = ""
        self.prefix = None

    def filterAcceptsRow(self, row, parent):
        if self.prefix is None:
            return True
        model = self.sourceModel()
        if QtCore.QDir.cleanPath(model.filePath(parent)) != self.root_path:
            return True
        index = model.index(row, 0, parent)
        return model.isDir(index) and model.fileName(index).startswith(self.prefix)


def build_file_tree(cfg: LayoutConfig) -> QtWidgets.QTreeView:
    model = QtWidgets.QFileSystemModel()
    # AllDirs: the name filters below apply to files only, not folder names.
    model.setFilter(QtCore.QDir.AllDirs | QtCore.QDir.Files | QtCore.QDir.Drives
                    | QtCore.QDir.NoDotAndDotDot)
    if cfg.tree_file_filters:
        model.setNameFilters(list(cfg.tree_file_filters))
        model.setNameFilterDisables(False)        # hide, rather than grey out
    model.sort(0)                                 # folders first, then by name
    proxy = TopLevelFilter()
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
    """Column names; dragging carries the selected names (one per line) as `mime_type`."""

    def __init__(self, mime_type: str):
        super().__init__()
        self.mime_type = mime_type

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


def filter_column_list(columns: QtWidgets.QListWidget, text: str) -> None:
    """Case-insensitive substring match; empty text shows everything."""
    needle = text.strip().lower()
    for i in range(columns.count()):
        item = columns.item(i)
        item.setHidden(needle not in item.text().lower())


def fill_column_list(columns: QtWidgets.QListWidget, search: QtWidgets.QLineEdit,
                     df: pl.DataFrame | None) -> None:
    columns.clear()
    for name, dtype in ({} if df is None else df.schema).items():
        item = QtWidgets.QListWidgetItem(name)
        item.setToolTip(str(dtype))
        columns.addItem(item)
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


def add_graph(graphs: GraphGrid, row: int, col: int) -> Graph:
    graph = Graph(None, graphs.grid.add_grid(row=row, col=col))
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


def match_axes(graphs: GraphGrid, graph: Graph) -> None:
    """Swap the graph's axes when its spec switches between 2D and 3D. A cell can't
    be emptied and reused, so a fresh cell takes the old one's grid slot."""
    three_d = is_3d(graph.spec.kind)
    if isinstance(graph.ax, vp.Axes3D) == three_d:
        return
    row, col = graph_position(graphs, graph)
    graphs.grid.remove_widget(graph.cell)
    graph.cell.parent = None
    graph.cell = graphs.grid.add_grid(row=row, col=col)
    attach_axes(graphs, graph, three_d)
    graphs.zoom_history[:] = [g for g in graphs.zoom_history if g is not graph]


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
    row = len(graphs.cells)
    graphs.cells.append([add_graph(graphs, row, col) for col in range(len(graphs.cells[0]))])
    graphs.canvas.update()


def add_graph_col(graphs: GraphGrid) -> None:
    col = len(graphs.cells[0])
    for row, cells_row in enumerate(graphs.cells):
        cells_row.append(add_graph(graphs, row, col))
    graphs.canvas.update()


def remove_graph_row(graphs: GraphGrid, row: int) -> None:
    relayout_graphs(graphs, removed=graphs.cells.pop(row))


def remove_graph_col(graphs: GraphGrid, col: int) -> None:
    relayout_graphs(graphs, removed=[cells_row.pop(col) for cells_row in graphs.cells])


def relayout_graphs(graphs: GraphGrid, removed: list[Graph]) -> None:
    """Drop `removed` from the canvas and re-place the rest at their new row/col."""
    for graph in removed:
        graphs.grid.remove_widget(graph.cell)
        graph.cell.parent = None
    for graph in all_graphs(graphs):
        graphs.grid.remove_widget(graph.cell)
    for r, cells_row in enumerate(graphs.cells):
        for c, graph in enumerate(cells_row):
            graphs.grid.add_widget(graph.cell, row=r, col=c)
            # add_widget gives the cell new size variables, but the cell is a Grid
            # whose own solver was built on the old ones; make it rebuild.
            graph.cell._need_solver_recreate = True
    graphs.zoom_history[:] = [g for g in graphs.zoom_history if g not in removed]
    graphs.canvas.update()


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


def fit_grid_to_canvas(graphs: GraphGrid, cfg: LayoutConfig) -> None:
    w, h = graphs.canvas.size
    left, top, right, bottom = cfg.graph_margins_px
    graphs.grid.pos = (left, top)
    graphs.grid.size = (max(w - left - right, 1), max(h - top - bottom, 1))


def build_graph_grid(cfg: LayoutConfig) -> GraphGrid:
    apply_vplot_params(cfg)
    canvas = scene.SceneCanvas(keys="interactive", bgcolor=cfg.graph_bgcolor, show=False,
                               dpi=vp.SCREEN_DPI)
    vp.skip_offscreen_swap(canvas)                # halves frame time on Intel GPUs
    # Placed by hand rather than central_widget.add_grid, so the x and y margins can differ.
    grid = scene.widgets.Grid(spacing=cfg.graph_spacing_px, parent=canvas.scene)
    graphs = GraphGrid(canvas, grid,
                       rubber_band=QtWidgets.QRubberBand(QtWidgets.QRubberBand.Rectangle,
                                                         canvas.native))
    vp.keep_ticks_synced(canvas, lambda: [g.ax for g in all_graphs(graphs)
                                          if isinstance(g.ax, vp.Axes)])
    if cfg.print_fps:
        canvas.measure_fps()
    canvas.events.resize.connect(lambda event: fit_grid_to_canvas(graphs, cfg))
    fit_grid_to_canvas(graphs, cfg)
    graphs.cells.append([add_graph(graphs, 0, 0)])
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
    ax.set_title(spec.title)
    ax.set_xlabel(spec.xlabel)
    ax.set_ylabel(spec.ylabel)
    if is_3d(spec.kind):
        ax.set_zlabel(spec.zlabel)
    for line, data in plot_data(spec, df, cfg):
        if spec.kind == "line":
            ax.plot(*data, color=line.color, label=line.label)
        else:                                     # scatter size is an area, in points^2
            ax.scatter(*data, s=spec.marker_size ** 2, color=line.color, label=line.label)
    ax.autoscale()
    # 3D axes have no slot below them, so their legend sits inside the plot.
    ax.legend(loc="upper right" if is_3d(spec.kind) else "below", fontsize=cfg.legend_fontsize)


def plot_data(spec: GraphSpec, df: pl.DataFrame | None, cfg: LayoutConfig):
    """(line, (x, y[, z]) arrays) for each line whose columns are all in df."""
    needed = [spec.x, spec.z] if is_3d(spec.kind) else [spec.x]
    if df is None or not all(c in df.columns for c in needed):
        return []
    x = df[spec.x].to_numpy()
    if is_3d(spec.kind):
        z = df[spec.z].to_numpy()
        return [(line, (x, df[line.column].to_numpy(), z))
                for line in spec.lines if line.column in df.columns]
    return [(line, (x, y_values(df[line.column], line.rolling, cfg)))
            for line in spec.lines if line.column in df.columns]


def y_values(column: pl.Series, rolling: int, cfg: LayoutConfig) -> np.ndarray:
    """The column, as a rolling mean over `rolling` rows when that's above 1. The
    ends average over the rows available, so there are no missing points."""
    if rolling > 1:
        column = column.cast(pl.Float64).rolling_mean(
            window_size=rolling, min_samples=1, center=cfg.rolling_center)
    return column.to_numpy()


def redraw_all(parts: "AppParts", cfg: LayoutConfig) -> None:
    for graph in all_graphs(parts.graphs):
        draw_graph(parts.graphs, graph, parts.state.df, cfg)
    parts.graphs.canvas.update()


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
def connect_graph_menu(parts: "AppParts", cfg: LayoutConfig) -> None:
    """A right click (press and release without moving) opens the menu for the
    graph under the mouse."""
    canvas = parts.graphs.canvas
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
        graph = graph_at(parts.graphs, event.pos)
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
        graphs.canvas.update()

    def clear():
        set_spec(parts, graph, GraphSpec(), cfg)

    def remove_row():
        if LAYOUT.require_confirm_on_graph_delete == True:
            if confirm(parts.window, f"Remove row {row + 1} ({len(graphs.cells[0])} graphs)?", cfg):
                push_undo(parts, cfg)
                remove_graph_row(graphs, row)
        else:
            push_undo(parts, cfg)
            remove_graph_row(graphs, row)

    def remove_col():
        if LAYOUT.require_confirm_on_graph_delete == True:
            if confirm(parts.window, f"Remove column {col + 1} ({len(graphs.cells)} graphs)?", cfg):
                push_undo(parts, cfg)
                remove_graph_col(graphs, col)
        else:
            push_undo(parts, cfg)
            remove_graph_col(graphs, col)

    menu = QtWidgets.QMenu(parts.window)
    menu.addAction(cfg.graph_menu_configure_text, lambda: configure_graph(parts, graph, cfg))
    menu.addAction(cfg.graph_menu_autoscale_text, autoscale)
    menu.addAction(cfg.graph_menu_undo_zoom_text, lambda: undo_zoom(graphs)).setEnabled(
        bool(graphs.zoom_history))
    menu.addAction(cfg.graph_menu_clear_text, clear)
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
        name, label, rolling = ((item, item, 1) if isinstance(item, str)
                                else (item.column, item.label, item.rolling))
        if name in shown or (df is not None and name not in numeric):
            continue
        shown.add(name)
        spec.lines.append(LineSpec(name, label, unused_line_color(cfg, spec.lines), rolling))
    if spec.x is None and spec.lines:
        spec.x = default_x(df, cfg) or cfg.default_x_column
        if spec.x == cfg.default_x_column and not spec.xlabel:
            spec.xlabel = cfg.default_xlabel
    set_spec(parts, graph, spec, cfg)


class ColumnDropFilter(QtCore.QObject):
    """Event filter on the graph canvas: accepts column drags over a graph."""

    def __init__(self, parts: "AppParts", cfg: LayoutConfig):
        super().__init__(parts.graphs.canvas.native)
        self.parts, self.cfg = parts, cfg

    def target(self, event) -> tuple[Graph | None, list[str]]:
        names = dropped_columns(event.mimeData(), self.cfg)
        pos = event.position()
        graph = graph_at(self.parts.graphs, (pos.x(), pos.y())) if names else None
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


def connect_column_drops(parts: "AppParts", cfg: LayoutConfig) -> None:
    native = parts.graphs.canvas.native
    native.setAcceptDrops(True)
    native.installEventFilter(ColumnDropFilter(parts, cfg))   # parented to native


# ---------------------------------------------------------------------------
# Graph dialog
# ---------------------------------------------------------------------------
class LineRow(NamedTuple):
    widget: QtWidgets.QWidget
    column: QtWidgets.QComboBox
    label: QtWidgets.QLineEdit
    rolling: QtWidgets.QSpinBox
    color: QtWidgets.QToolButton


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
    rolling.setPrefix("avg ")
    rolling.setToolTip(cfg.rolling_tooltip)
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
    layout.addWidget(color)
    layout.addWidget(remove)
    row = LineRow(widget, combo, label, rolling, color)
    remove.clicked.connect(lambda: on_remove(row))
    return row


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
    xlabel = QtWidgets.QLineEdit(spec.xlabel)
    if new_graph and not spec.xlabel and x_combo.currentText() == cfg.default_x_column:
        xlabel.setText(cfg.default_xlabel)
    ylabel = QtWidgets.QLineEdit(spec.ylabel)
    z_combo = build_column_combo(columns, spec.z)
    zlabel = QtWidgets.QLineEdit(spec.zlabel)
    form = QtWidgets.QFormLayout()
    form.addRow("Title", title)
    form.addRow("Graph type", kind)
    form.addRow("Point size", marker_size)
    form.addRow("X column", x_combo)
    form.addRow("X label", xlabel)
    form.addRow("Y label", ylabel)
    form.addRow("Z column", z_combo)
    form.addRow("Z label", zlabel)

    def show_kind_rows():
        """Point size only for scatter, Z only for 3D; the type itself is advanced."""
        key = kind.currentData()
        form.setRowVisible(kind, feature("graph_type"))
        form.setRowVisible(marker_size, key != "line")
        form.setRowVisible(z_combo, is_3d(key))
        form.setRowVisible(zlabel, is_3d(key))
        for row in rows:                          # rolling mean is 2D only
            row.rolling.setVisible(not is_3d(key))

    rows: list[LineRow] = []
    rows_box = QtWidgets.QWidget()
    rows_layout = QtWidgets.QVBoxLayout(rows_box)
    rows_layout.addStretch(1)                     # rows are inserted above this
    scroll = QtWidgets.QScrollArea()
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
    buttons.accepted.connect(dialog.accept)
    buttons.rejected.connect(dialog.reject)

    layout = QtWidgets.QVBoxLayout(dialog)
    layout.addLayout(form)
    layout.addWidget(QtWidgets.QLabel("Y lines  (column / legend label / rolling mean / color)"))
    layout.addWidget(scroll, 1)
    layout.addWidget(add_button, 0, QtCore.Qt.AlignLeft)
    layout.addWidget(buttons)

    if dialog.exec() != QtWidgets.QDialog.Accepted:
        return None

    def valid(name):
        # Without a loaded file anything typed is kept, so specs can be set up early.
        return bool(name) and (not columns or name in columns)

    lines = [LineSpec(r.column.currentText(), r.label.text() or r.column.currentText(),
                      r.color.property("color"), r.rolling.value())
             for r in rows if valid(r.column.currentText())]
    x, z = x_combo.currentText(), z_combo.currentText()
    return GraphSpec(x if valid(x) else None, lines, xlabel.text(), ylabel.text(), title.text(),
                     kind.currentData(), marker_size.value(), z if valid(z) else None,
                     zlabel.text())


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
# Folder selection
# ---------------------------------------------------------------------------
def apply_data_dir(parts: AppParts, path: str, cfg: LayoutConfig) -> None:
    """Point the app at `path`: tooltip, file tree, and a warning if misnamed."""
    parts.select_button.setToolTip(f"Data folder:\n{path}")
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


def build_select_button(cfg: LayoutConfig) -> QtWidgets.QPushButton:
    button = QtWidgets.QPushButton(cfg.select_button_text)
    button.setToolTip("No data folder selected")
    button.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Ignored)
    return button


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
    column_page, column_search, column_list = build_column_page(cfg)

    stack = QtWidgets.QStackedWidget()
    stack.insertWidget(FILES_PAGE, tree)
    stack.insertWidget(COLUMNS_PAGE, column_page)

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
    return SideBar(frame, files_button, columns_button, stack, tree, column_search, column_list)


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


def load_advanced(cfg: LayoutConfig) -> bool:
    return _settings(cfg).value(cfg.settings_advanced_key, False, type=bool)


def load_script(cfg: LayoutConfig) -> str:
    return _settings(cfg).value(cfg.settings_script_key, "", type=str)


def set_advanced(parts: AppParts, on: bool, cfg: LayoutConfig) -> None:
    """Advanced-mode switch: remembered between runs. Menus and dialogs check it when
    they open; the script button is shown/hidden here."""
    parts.state.advanced = on
    _settings(cfg).setValue(cfg.settings_advanced_key, on)
    parts.script_button.setVisible(feature_on(parts.state, "script", cfg))


def build_window(cfg: LayoutConfig = LAYOUT) -> AppParts:
    window = QtWidgets.QMainWindow()
    window.setWindowTitle(cfg.window_title)
    window.resize(*cfg.window_size)

    top_bar = build_top_bar(cfg)
    select_button = build_select_button(cfg)
    file_label = build_file_label(cfg)
    top_bar.layout().addWidget(select_button)
    top_bar.layout().addSpacing(12)
    top_bar.layout().addWidget(file_label)
    top_bar.layout().addStretch(1)
    script_button = QtWidgets.QPushButton(cfg.script_button_text)
    script_button.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Ignored)
    advanced_switch = ToggleSwitch(cfg)
    advanced_label = QtWidgets.QLabel(cfg.advanced_text)
    advanced_label.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Ignored)
    top_bar.layout().addWidget(script_button)
    top_bar.layout().addSpacing(12)
    top_bar.layout().addWidget(advanced_label)
    top_bar.layout().addWidget(advanced_switch)

    side = build_side_bar(cfg)
    graphs = build_graph_grid(cfg)
    before_change = lambda: push_undo(parts, cfg)  # parts is bound below, before any click
    splitter = build_splitter(side.frame, build_graph_area(graphs, before_change, cfg), cfg)

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

    parts = AppParts(window, top_bar, side, splitter, graphs, select_button, file_label,
                     script_button, AppState(script=load_script(cfg)))
    select_button.clicked.connect(lambda: choose_data_dir(parts, cfg))
    script_button.clicked.connect(lambda: open_script_window(parts, cfg))
    side.tree.clicked.connect(lambda index: on_tree_clicked(side.tree, index))
    side.tree.doubleClicked.connect(lambda index: on_tree_double_clicked(parts, index, cfg))
    connect_graph_menu(parts, cfg)
    connect_column_drops(parts, cfg)
    # Window-wide; a focused text box still gets Ctrl+Z / Ctrl+Y for its own text first.
    undo = QtGui.QShortcut(QtGui.QKeySequence("Ctrl+Z"), window)
    undo.activated.connect(lambda: step_history(parts, True, cfg))
    redo = QtGui.QShortcut(QtGui.QKeySequence("Ctrl+Y"), window)
    redo.activated.connect(lambda: step_history(parts, False, cfg))

    advanced_switch.setChecked(load_advanced(cfg))
    set_advanced(parts, advanced_switch.isChecked(), cfg)
    advanced_switch.toggled.connect(lambda on: set_advanced(parts, on, cfg))
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
