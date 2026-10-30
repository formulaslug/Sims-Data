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
"""

import os
import sys
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np
import polars as pl
from PySide6 import QtCore, QtWidgets
from vispy import app as vispy_app
from vispy import scene

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

    graph_grid_margin_px: int = 6
    plus_strip_px: int = 18                       # thickness of the "+" strips

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
class GraphGrid:
    """One SceneCanvas holding a rows x cols grid of vplot Axes."""
    canvas: scene.SceneCanvas
    grid: scene.widgets.Grid
    axes: list[list[vp.Axes]] = field(default_factory=list)


@dataclass
class AppState:
    """Things that change while the app runs."""
    df: pl.DataFrame | None = None
    df_path: str | None = None


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


def load_data_file(parts: AppParts, path: str, cfg: LayoutConfig) -> None:
    QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
    try:
        df = read(path)
    except (pl.exceptions.PolarsError, OSError) as e:
        show_toast(parts, f"Could not read {os.path.basename(path)}: {e}", cfg)
        return
    finally:
        QtWidgets.QApplication.restoreOverrideCursor()
    parts.state.df, parts.state.df_path = df, path
    parts.file_label.setText(os.path.basename(path))
    parts.file_label.setToolTip(f"{path}\n{df.height:,} rows x {df.width} columns")
    fill_column_list(parts.side, df)


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
    proxy.root_path = QtCore.QDir.cleanPath(path)
    proxy.prefix = cfg.data_subfolder_prefix if is_data_folder(path, cfg) else None
    proxy.invalidateFilter()
    model.setRootPath(path)
    tree.setRootIndex(proxy.mapFromSource(model.index(path)))


# ---------------------------------------------------------------------------
# Column list
# ---------------------------------------------------------------------------
def build_column_page(cfg: LayoutConfig) -> tuple[QtWidgets.QWidget, QtWidgets.QLineEdit,
                                                  QtWidgets.QListWidget]:
    page = QtWidgets.QWidget()
    search = QtWidgets.QLineEdit()
    search.setPlaceholderText(cfg.column_search_placeholder)
    search.setClearButtonEnabled(True)
    search.setStyleSheet(f"background-color: {cfg.input_color}; border: none; padding: 4px;")
    columns = QtWidgets.QListWidget()
    columns.setFrameShape(QtWidgets.QFrame.NoFrame)
    columns.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
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


def fill_column_list(side: SideBar, df: pl.DataFrame) -> None:
    side.column_list.clear()
    for name, dtype in df.schema.items():
        item = QtWidgets.QListWidgetItem(name)
        item.setToolTip(str(dtype))
        side.column_list.addItem(item)
    filter_column_list(side.column_list, side.column_search.text())


# ---------------------------------------------------------------------------
# Graph grid
# ---------------------------------------------------------------------------
def add_graph(graphs: GraphGrid, row: int, col: int) -> vp.Axes:
    cell = graphs.grid.add_grid(row=row, col=col)
    return vp.Axes(None, cell)


def add_graph_row(graphs: GraphGrid) -> None:
    row = len(graphs.axes)
    graphs.axes.append([add_graph(graphs, row, col) for col in range(len(graphs.axes[0]))])
    graphs.canvas.update()


def add_graph_col(graphs: GraphGrid) -> None:
    col = len(graphs.axes[0])
    for row, axes_row in enumerate(graphs.axes):
        axes_row.append(add_graph(graphs, row, col))
    graphs.canvas.update()


def build_graph_grid(cfg: LayoutConfig) -> GraphGrid:
    canvas = scene.SceneCanvas(keys="interactive", bgcolor=cfg.graph_bgcolor, show=False,
                               dpi=vp.SCREEN_DPI)
    grid = canvas.central_widget.add_grid(margin=cfg.graph_grid_margin_px)
    graphs = GraphGrid(canvas, grid)
    graphs.axes.append([add_graph(graphs, 0, 0)])
    return graphs


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


def build_graph_area(graphs: GraphGrid, cfg: LayoutConfig) -> QtWidgets.QWidget:
    """Graph canvas with a "+" strip on the right (columns) and bottom (rows)."""
    area = QtWidgets.QWidget()
    layout = QtWidgets.QGridLayout(area)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(0)
    add_col = build_plus_strip(vertical=True, cfg=cfg)
    add_row = build_plus_strip(vertical=False, cfg=cfg)
    add_col.clicked.connect(lambda: add_graph_col(graphs))
    add_row.clicked.connect(lambda: add_graph_row(graphs))
    layout.addWidget(graphs.canvas.native, 0, 0)
    layout.addWidget(add_col, 0, 1)
    layout.addWidget(add_row, 1, 0)
    return area


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

    side = build_side_bar(cfg)
    graphs = build_graph_grid(cfg)
    splitter = build_splitter(side.frame, build_graph_area(graphs, cfg), cfg)

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

    parts = AppParts(window, top_bar, side, splitter, graphs,
                     select_button, file_label, AppState())
    select_button.clicked.connect(lambda: choose_data_dir(parts, cfg))
    side.tree.clicked.connect(lambda index: on_tree_clicked(side.tree, index))
    side.tree.doubleClicked.connect(lambda index: on_tree_double_clicked(parts, index, cfg))
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
