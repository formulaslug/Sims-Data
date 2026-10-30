"""
vplot -- a small matplotlib.pyplot-style wrapper around VisPy (GPU/OpenGL).

    import numpy as np
    import vplot as vp

    x = np.linspace(0, 10, 1_000_000)
    fig, (ax1, ax2) = vp.subplots(2, 1, sharex=True)
    ax1.plot(x, np.sin(x), label="sin")
    ax1.plot(x, np.cos(x), "r", label="cos")
    ax1.legend()
    ax1.set_ylabel("amplitude")
    sc = ax2.scatter(x[::1000], np.sin(x[::1000]), c=x[::1000], cmap="viridis")
    fig.colorbar(sc, ax=ax2, label="x")
    vp.show()

Supported (pyplot-like):
    figure, subplots, subplot, gcf, gca, show, savefig, close
    plot (2D and 3D, fmt strings like 'r--o'), scatter (2D and 3D, c/cmap/vmin/vmax/s)
    xlabel/ylabel/zlabel, title, xlim/ylim/zlim, legend, grid, colorbar, twinx,
    axhline/axvline, text, set_aspect('equal'), tick_params(labelsize=)
    Font sizes: vp.rcParams["font.size"] = 14 (before creating figures), or per
    call: set_xlabel("x", fontsize=16), set_title(..., fontsize=), legend(fontsize=).
    Closing a window forgets that figure, so show() only shows new ones.
    Artists returned by plot/scatter have set_data(...) for fast updates.
    fig.add_slider(...) gives matplotlib-Slider-like controls (Qt backends), and the
    Left/Right arrow keys step the active slider on any backend (Up/Down switch slider).

Interaction: drag to pan, scroll to zoom (2D). In 3D, drag to rotate,
shift+drag to pan, scroll to zoom.

Known differences from matplotlib:
    * Dashed/dotted line styles are drawn solid (VisPy lines have no dash support).
    * Line widths > 1 px need method='agg' (the default for lines up to 100k points);
      very large lines use method='gl', which some drivers clamp to 1 px.
    * No log axes, no imshow/contour/errorbar (yet).
    * twinx() continues the parent's color cycle, and parent.legend() includes
      the twin's labeled lines.
"""

from __future__ import annotations

import math
import re
import warnings

import numpy as np

from vispy import app, scene
from vispy.color import Color, Colormap, get_colormap
from vispy.visuals.transforms import STTransform

__all__ = [
    "Figure", "Axes", "Axes3D", "Slider",
    "figure", "subplots", "subplot", "gcf", "gca", "show", "savefig", "close",
    "plot", "scatter", "xlabel", "ylabel", "zlabel", "title", "xlim", "ylim",
    "zlim", "legend", "grid", "colorbar", "twinx", "axhline", "axvline", "text",
    "tick_params", "rcParams",
]

# --------------------------------------------------------------------------
# style helpers
# --------------------------------------------------------------------------

_CYCLE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
          "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
_SHORT_COLORS = {
    "b": (0.0, 0.0, 1.0), "g": (0.0, 0.5, 0.0), "r": (1.0, 0.0, 0.0),
    "c": (0.0, 0.75, 0.75), "m": (0.75, 0.0, 0.75), "y": (0.75, 0.75, 0.0),
    "k": (0.0, 0.0, 0.0), "w": (1.0, 1.0, 1.0),
}
_MARKERS = {
    "o": "disc", ".": "disc", ",": "square", "s": "square", "^": "triangle_up",
    "v": "triangle_down", "*": "star", "+": "cross", "x": "x", "d": "diamond",
    "D": "diamond", "|": "vbar", "_": "hbar",
}
_LINESTYLES = ("--", "-.", "-", ":")
SCREEN_DPI = 96.0         # canvas dpi; HiDPI scaling is applied by VisPy's pixel_scale.
                          # Leaving VisPy's detected dpi (which already includes Windows
                          # display scaling) scales text twice on HiDPI screens.
PT = SCREEN_DPI / 72.0    # points -> pixels
_FG = (0.0, 0.0, 0.0, 1.0)  # axis / text color

# Font sizes in points, like matplotlib's rcParams. Change before creating a
# figure, e.g. vp.rcParams["font.size"] = 14. Entries left as None are derived
# from "font.size". Text keeps a fixed size when the window is resized.
rcParams = {
    "font.size": 12.0,
    "axes.titlesize": None,      # default 1.2 x font.size
    "axes.labelsize": None,      # default 1.0 x font.size
    "tick.labelsize": None,      # default 0.85 x font.size
    "legend.fontsize": None,     # default 0.85 x font.size
    # Gap in pixels between the axis line and its tick labels.
    "xtick.pad": None,           # default 1.0 x tick label height
    "ytick.pad": None,           # default 0.6 x tick.labelsize + 2
}
_REL = {"axes.titlesize": 1.2, "axes.labelsize": 1.0,
        "tick.labelsize": 0.85, "legend.fontsize": 0.85}


def _tick_pad(orientation, size):
    """Axis line -> tick label gap (px) for tick labels of `size` points."""
    key = "ytick.pad" if orientation in ("left", "right") else "xtick.pad"
    v = rcParams.get(key)
    if v is not None:
        return float(v)
    return int(size * 0.6) + 2 if key == "ytick.pad" else size * PT


def _fs(key):
    v = rcParams.get(key)
    return float(v) if v is not None else rcParams["font.size"] * _REL[key]


def _set_text_size(obj, size, *attrs):
    """Best-effort font-size change on a VisPy widget/visual's inner TextVisual."""
    for name in attrs:
        if name == "self":
            target = obj
        else:
            target = getattr(obj, name, None)
        if target is not None and hasattr(type(target), "font_size"):
            target.font_size = size
            return True
    return False


def _set_axis_tick_size(aw, size):
    ax = aw.axis
    if isinstance(getattr(type(ax), "tick_font_size", None), property):
        ax.tick_font_size = size
    elif not _set_text_size(ax, size, "_text"):
        _warn_once("ticksize", "vplot: this VisPy version can't resize tick labels "
                               "after creation; set vp.rcParams before plotting")


def _rgba(c, alpha=None):
    """Any matplotlib-ish color spec -> RGBA float array."""
    if isinstance(c, str):
        if c in _SHORT_COLORS:
            c = _SHORT_COLORS[c]
        else:
            m = re.fullmatch(r"C(\d)", c)
            if m:
                c = _CYCLE[int(m.group(1)) % len(_CYCLE)]
    rgba = np.array(Color(c).rgba, dtype=np.float32)
    if alpha is not None:
        rgba[3] = alpha
    return rgba


def _parse_fmt(fmt):
    """'r--o' -> (color, marker, linestyle). Missing parts are None."""
    color = marker = ls = None
    i = 0
    while i < len(fmt):
        two = fmt[i:i + 2]
        if two in ("--", "-."):
            ls = two
            i += 2
            continue
        ch = fmt[i]
        if ch in ("-", ":"):
            ls = ch
        elif ch in _SHORT_COLORS:
            color = ch
        elif ch == "C" and i + 1 < len(fmt) and fmt[i + 1].isdigit():
            color = fmt[i:i + 2]
            i += 1
        elif ch in _MARKERS:
            marker = ch
        else:
            raise ValueError(f"Unrecognized format string {fmt!r}")
        i += 1
    if marker is not None and ls is None:
        ls = ""          # 'o' alone means markers only, like matplotlib
    return color, marker, ls


def _parse_plot_args(args, ndim):
    """Split plot(*args) into [(arrays, fmt), ...] groups like matplotlib."""
    groups, i = [], 0
    while i < len(args):
        arrs = []
        while i < len(args) and not isinstance(args[i], str) and len(arrs) < ndim:
            arrs.append(args[i])
            i += 1
        fmt = ""
        if i < len(args) and isinstance(args[i], str):
            fmt = args[i]
            i += 1
        if not arrs:
            raise ValueError("plot() got a format string with no data")
        groups.append((arrs, fmt))
    return groups


def _nice_ticks(lo, hi, n=5):
    if hi < lo:
        lo, hi = hi, lo
    span = hi - lo
    if span <= 0 or not np.isfinite(span):
        return np.array([lo])
    raw = span / n
    mag = 10 ** math.floor(math.log10(raw))
    for m in (1, 2, 2.5, 5, 10):
        step = m * mag
        if span / step <= n:
            break
    start = math.ceil(lo / step) * step
    ticks = np.arange(start, hi + step * 1e-6, step)
    ticks[np.abs(ticks) < step * 1e-9] = 0.0
    return ticks


def _pad(lo, hi, frac=0.05):
    if hi == lo:
        d = abs(lo) * frac or 0.5
    else:
        d = (hi - lo) * frac
    return lo - d, hi + d


def _stack(x, y, z=None):
    x = np.asarray(x, dtype=np.float32).ravel()
    y = np.asarray(y, dtype=np.float32).ravel()
    cols = [x, y] if z is None else [x, y, np.asarray(z, dtype=np.float32).ravel()]
    n = min(len(c) for c in cols)
    if any(len(c) != n for c in cols):
        raise ValueError("x, y (and z) must have the same length")
    return np.column_stack(cols)


def _cmap_lut(cmap, n=256):
    """Return (vispy Colormap, (n,4) lookup table)."""
    if isinstance(cmap, Colormap):
        cm = cmap
    else:
        cm = None
        try:
            cm = get_colormap(cmap)
            lut = np.asarray(cm.map(np.linspace(0, 1, n)), dtype=np.float32).reshape(-1, 4)
            return cm, lut
        except Exception:
            pass
        try:  # fall back to matplotlib's colormaps if it's installed
            import matplotlib
            m = matplotlib.colormaps[cmap]
            cm = Colormap(m(np.linspace(0, 1, 64)))
        except Exception as exc:
            raise ValueError(f"Unknown colormap {cmap!r}") from exc
    lut = np.asarray(cm.map(np.linspace(0, 1, n)), dtype=np.float32).reshape(-1, 4)
    return cm, lut


def _apply_lut(values, lut, clim):
    lo, hi = clim
    v = np.asarray(values, dtype=np.float64)
    t = (v - lo) / (hi - lo) if hi != lo else np.zeros_like(v)
    bad = ~np.isfinite(t)
    idx = np.clip(np.nan_to_num(t) * (len(lut) - 1), 0, len(lut) - 1).round().astype(int)
    out = lut[idx].copy()
    out[bad, 3] = 0.0
    return out


def _set_markers(m, pos, size, face, edge, edge_width, symbol):
    kw = dict(pos=pos, size=size, face_color=face, edge_color=edge, edge_width=edge_width)
    try:
        m.set_data(symbol=symbol, **kw)
    except TypeError:          # newer VisPy: symbol is a property
        m.set_data(**kw)
        m.symbol = symbol


def _overlay(visual, order):
    visual.order = order
    visual.set_gl_state("translucent", depth_test=False)


_warned = set()


def _warn_once(key, msg):
    if key not in _warned:
        _warned.add(key)
        warnings.warn(msg, stacklevel=3)


# --------------------------------------------------------------------------
# artists
# --------------------------------------------------------------------------

class _Artist:
    label = None

    def bounds(self):
        if self._pos is None or len(self._pos) == 0:
            return None
        lo = np.nanmin(self._pos, axis=0)
        hi = np.nanmax(self._pos, axis=0)
        if not np.all(np.isfinite(lo)):
            return None
        return np.column_stack([lo, hi])

    def _visuals(self):
        return []

    def set_visible(self, b):
        for v in self._visuals():
            v.visible = bool(b)

    def remove(self):
        for v in self._visuals():
            v.parent = None
        if self in self._ax._artists:
            self._ax._artists.remove(self)

    def _set_order(self, n):
        for v in self._visuals():
            v.order = n


class Line(_Artist):
    """Returned by Axes.plot. Update with set_data(x, y[, z])."""

    def __init__(self, ax, pos, color, width, marker, markersize, linestyle,
                 label, method):
        self._ax, self._pos, self.label = ax, pos, label
        self.color, self.width = color, width
        self.marker, self.markersize = marker, markersize
        self._line = self._markers = None
        if linestyle not in ("", " ", "None", "none"):
            if linestyle in ("--", "-.", ":"):
                _warn_once("dash", "vplot: dashed/dotted lines are drawn solid")
            if method is None:
                method = "gl" if (pos.shape[1] == 3 or len(pos) > 100_000) else "agg"
            self._method = method
            self._line = scene.visuals.Line(pos=pos, color=color, width=self._gl_width(width),
                                            method=method, parent=ax._data_parent)
        if marker:
            self._markers = scene.visuals.Markers(parent=ax._data_parent)
            self._update_markers()

    def _gl_width(self, w):
        # many OpenGL drivers reject widths > 1 for raw GL lines
        return w if self._method == "agg" else 1.0

    def _update_markers(self):
        _set_markers(self._markers, self._pos, self.markersize, self.color,
                     self.color, 0.0, _MARKERS[self.marker])

    def _visuals(self):
        return [v for v in (self._line, self._markers) if v is not None]

    def set_data(self, x, y=None, z=None):
        if y is None:
            y, x = x, np.arange(len(x))
        self._pos = _stack(x, y, z)
        if self._line is not None:
            self._line.set_data(pos=self._pos)
        if self._markers is not None:
            self._update_markers()

    def set_ydata(self, y):
        z = self._pos[:, 2] if self._pos.shape[1] == 3 else None
        self.set_data(self._pos[:, 0], y, z)

    def set_color(self, c):
        self.color = _rgba(c)
        if self._line is not None:
            self._line.set_data(color=self.color)
        if self._markers is not None:
            self._update_markers()

    def set_linewidth(self, w):
        self.width = w * PT
        if self._line is not None:
            self._line.set_data(width=self._gl_width(self.width))


class Scatter(_Artist):
    """Returned by Axes.scatter. Update with set_data(x, y, z=None, c=None, s=None)."""

    def __init__(self, ax, pos, c, s, cmap, vmin, vmax, marker, alpha,
                 edgecolors, linewidths, label):
        self._ax, self._pos, self.label = ax, pos, label
        self._alpha, self._symbol = alpha, _MARKERS.get(marker, marker)
        self._edge = None if edgecolors in (None, "none", "face") else _rgba(edgecolors)
        self._edge_width = 0.0 if self._edge is None else (linewidths or 1.0) * PT
        self._cmap, self._lut = _cmap_lut(cmap or "viridis")
        self._colorbars = []
        self._values = None
        self._solid = None
        self._sizes = self._size_px(s)
        self._vmin, self._vmax = vmin, vmax
        self._set_c(c, first=True)
        self._m = scene.visuals.Markers(parent=ax._data_parent)
        self._push()

    @staticmethod
    def _size_px(s):
        s = 36.0 if s is None else s   # matplotlib default area, pt^2
        return np.sqrt(np.asarray(s, dtype=np.float32)) * PT

    def _set_c(self, c, first=False):
        n = len(self._pos)
        if c is None:
            if first:
                self._solid = _rgba(self._ax._next_color(), self._alpha)
            return
        arr = None if isinstance(c, str) else np.asarray(c)
        if arr is not None and arr.ndim == 1 and arr.size == n and np.issubdtype(arr.dtype, np.number):
            self._values, self._solid = arr.astype(np.float64), None
            if first or self._clim is None:
                lo = self._vmin if self._vmin is not None else np.nanmin(arr)
                hi = self._vmax if self._vmax is not None else np.nanmax(arr)
                self._clim = (float(lo), float(hi))
        elif arr is not None and arr.ndim == 2 and arr.shape[0] == n and arr.shape[1] in (3, 4):
            rgba = np.ones((n, 4), np.float32)
            rgba[:, :arr.shape[1]] = arr
            if self._alpha is not None:
                rgba[:, 3] = self._alpha
            self._values, self._solid = None, rgba
        else:
            self._values, self._solid = None, _rgba(c, self._alpha)

    _clim = None

    @property
    def clim(self):
        return self._clim if self._clim is not None else (0.0, 1.0)

    def _face(self):
        if self._values is not None:
            rgba = _apply_lut(self._values, self._lut, self.clim)
            if self._alpha is not None:
                rgba[:, 3] *= self._alpha
            return rgba
        return self._solid

    def _push(self):
        face = self._face()
        edge = self._edge if self._edge is not None else face
        _set_markers(self._m, self._pos, self._sizes, face, edge,
                     self._edge_width, self._symbol)

    def _visuals(self):
        return [self._m]

    def set_data(self, x=None, y=None, z=None, c=None, s=None):
        if x is not None:
            if y is None:
                raise ValueError("set_data needs both x and y")
            if z is None and self._pos.shape[1] == 3:
                raise ValueError("this is a 3D scatter; pass z too")
            self._pos = _stack(x, y, z)
        if s is not None:
            self._sizes = self._size_px(s)
        if c is not None:
            self._set_c(c)
        self._push()

    # matplotlib-compatible aliases
    def set_offsets(self, xy):
        xy = np.asarray(xy)
        self.set_data(xy[:, 0], xy[:, 1])

    def set_array(self, c):
        self.set_data(c=c)

    def set_sizes(self, s):
        self.set_data(s=s)

    def set_clim(self, vmin=None, vmax=None):
        if isinstance(vmin, (tuple, list)):
            vmin, vmax = vmin
        lo, hi = self.clim
        self._clim = (lo if vmin is None else float(vmin), hi if vmax is None else float(vmax))
        for cb in self._colorbars:
            cb.clim = self._clim
        self._push()

    def set_cmap(self, cmap):
        self._cmap, self._lut = _cmap_lut(cmap)
        for cb in self._colorbars:
            cb.cmap = self._cmap
        self._push()


# --------------------------------------------------------------------------
# axes
# --------------------------------------------------------------------------

def _axis_widget(orientation, size):
    kw = dict(axis_color=_FG, tick_color=_FG, text_color=_FG, tick_font_size=size,
              tick_label_margin=_tick_pad(orientation, size), major_tick_length=6,
              minor_tick_length=3, tick_width=1, axis_width=1)
    try:
        aw = scene.AxisWidget(orientation=orientation, **kw)
    except TypeError:   # older VisPy without some of these options
        aw = scene.AxisWidget(orientation=orientation, axis_color=_FG,
                              tick_color=_FG, text_color=_FG)
        _set_axis_tick_size(aw, size)
    _size_axis_slot(aw, size)
    return aw


def _link_axis(aw, view):
    """link_view, plus recompute ticks when the axis itself is resized/moved.
    VisPy only recomputes on camera changes, so after a grid re-layout the
    ticks were left mapped with the axis' old geometry."""
    aw.link_view(view)
    refresh = lambda event=None: aw._view_changed()
    aw.events.resize.connect(refresh)
    aw.events.transform_change.connect(refresh)


def _size_axis_slot(aw, size):
    """Give a tick axis enough room for labels of `size` points."""
    px = size * PT
    pad = _tick_pad(aw.orientation, size)
    if aw.orientation in ("left", "right"):
        aw.width_max = int(pad + px * 3.6 + 10)
    else:
        aw.height_max = int(pad + px * 1.4 + 10)


def _label(text="", rotation=0.0, size=12.0):
    return scene.Label(text, rotation=rotation, color=_FG, font_size=size)


def _set_label(lbl, text, fontsize, vertical):
    lbl.text = text
    if fontsize is not None:
        _set_text_size(lbl, fontsize, "_text_visual", "_text")
        _size_label_slot(lbl, fontsize, vertical)


def _size_label_slot(lbl, size, vertical):
    px = size * PT
    if vertical:
        lbl.width_max = int(px * 1.8 + 4)
    else:
        lbl.height_max = int(px * 1.8 + 4)


class _AxesBase:
    """Shared plotting logic. Subclasses define _data_parent, view, _ndim."""

    def __init__(self, fig):
        self.figure = fig
        self._artists = []
        self._twins = []
        self._legend_node = None
        self._legend_kw = None

    # ---- color cycle / bookkeeping ----
    def _next_color(self):
        root = getattr(self, "_parent", None) or self
        c = _CYCLE[root._cycle_i % len(_CYCLE)] if hasattr(root, "_cycle_i") else _CYCLE[0]
        root._cycle_i = getattr(root, "_cycle_i", 0) + 1
        return c

    def _all_artists(self):
        out = list(self._artists)
        for t in self._twins:
            out += t._artists
        return out

    def _add(self, artist):
        self._artists.append(artist)
        artist._set_order(len(self._all_artists()))
        self._autoscale()
        if self._legend_kw is not None:
            self.legend(**self._legend_kw)
        return artist

    # ---- plotting ----
    def plot(self, *args, color=None, c=None, linewidth=None, lw=None,
             linestyle=None, ls=None, marker=None, markersize=None, ms=None,
             label=None, alpha=None, method=None):
        nd = self._ndim
        lines = []
        for arrs, fmt in _parse_plot_args(args, nd):
            fcolor, fmarker, fls = _parse_fmt(fmt)
            if nd == 2:
                if len(arrs) == 1:
                    y = np.asarray(arrs[0])
                    x = np.arange(y.shape[0])
                else:
                    x, y = np.asarray(arrs[0]), np.asarray(arrs[1])
                ys = [y] if y.ndim == 1 else [y[:, j] for j in range(y.shape[1])]
                pos_list = [_stack(x, yy) for yy in ys]
            else:
                if len(arrs) != 3:
                    raise ValueError("3D plot needs x, y, z")
                pos_list = [_stack(*arrs)]
            for pos in pos_list:
                col = color or c or fcolor or self._next_color()
                mk = marker if marker is not None else fmarker
                lstyle = linestyle or ls or (fls if fls is not None else "-")
                w = (linewidth or lw or 1.5) * PT
                msz = (markersize or ms or (3.0 if mk == "." else 6.0)) * PT
                line = Line(self, pos, _rgba(col, alpha), w, mk, msz, lstyle,
                            label, method)
                lines.append(self._add(line))
        return lines

    def scatter(self, x, y, *zs, s=None, c=None, color=None, cmap=None,
                vmin=None, vmax=None, marker="o", alpha=None, edgecolors=None,
                linewidths=None, label=None):
        if self._ndim == 3:
            if not zs:
                raise ValueError("3D scatter needs x, y, z")
            pos = _stack(x, y, zs[0])
        else:
            pos = _stack(x, y)
        if c is None:
            c = color
        sc = Scatter(self, pos, c, s, cmap, vmin, vmax, marker, alpha,
                     edgecolors, linewidths, label)
        return self._add(sc)

    def text(self, x, y, *args, color="black", fontsize=9, ha="left", va="center"):
        if self._ndim == 3:
            z, s = args
            pos = (x, y, z)
        else:
            (s,) = args
            pos = (x, y)
        t = scene.visuals.Text(s, pos=pos, color=_rgba(color), font_size=fontsize,
                               anchor_x=ha, anchor_y=va, parent=self._data_parent)
        return t

    # ---- legend (drawn in widget pixel coordinates) ----
    def legend(self, loc="upper right", fontsize=None):
        if fontsize is None:
            fontsize = _fs("legend.fontsize")
        self._legend_kw = dict(loc=loc, fontsize=fontsize)
        view = self.view
        if self._legend_node is not None:
            self._legend_node.parent = None
            self._legend_node = None
        items = [a for a in self._all_artists() if a.label and not str(a.label).startswith("_")]
        if not items:
            return None
        node = scene.Node(parent=view)
        node.transform = STTransform()
        lh, pad = fontsize * PT * 1.6, 6.0
        for i, a in enumerate(items):
            yc = pad + lh * (i + 0.5)
            if isinstance(a, Line):
                if a._line is not None:
                    ln = scene.visuals.Line(pos=np.array([[pad + 2, yc], [pad + 24, yc]], np.float32),
                                            color=a.color, width=max(a.width, 1), method="agg",
                                            parent=node)
                    _overlay(ln, 1001)
                if a._markers is not None:
                    mk = scene.visuals.Markers(parent=node)
                    _set_markers(mk, np.array([[pad + 13, yc]], np.float32), a.markersize,
                                 a.color, a.color, 0.0, _MARKERS[a.marker])
                    _overlay(mk, 1002)
            else:
                face = a._face()
                face = face[len(face) // 2] if np.ndim(face) == 2 and len(face) else face
                mk = scene.visuals.Markers(parent=node)
                _set_markers(mk, np.array([[pad + 13, yc]], np.float32), 8.0,
                             np.atleast_2d(face), np.atleast_2d(face), 0.0, a._symbol)
                _overlay(mk, 1002)
            t = scene.visuals.Text(str(a.label), pos=(pad + 32, yc), color=_FG,
                                   font_size=fontsize, anchor_x="left", anchor_y="center",
                                   parent=node)
            _overlay(t, 1003)
        w = pad + 32 + max(len(str(a.label)) for a in items) * fontsize * 0.85 + pad
        h = lh * len(items) + 2 * pad
        bg = scene.visuals.Rectangle(center=(w / 2, h / 2), width=w, height=h,
                                     color=(1, 1, 1, 0.85), border_color=(0.6, 0.6, 0.6, 1),
                                     parent=node)
        _overlay(bg, 1000)
        self._legend_node = node

        def place(event=None):
            vw, vh = view.size
            x = 8 if "left" in loc else vw - w - 8
            y = 8 if "upper" in loc else vh - h - 8
            node.transform.translate = (x, y)

        old = getattr(self, "_legend_place", None)
        if old is not None:
            view.events.resize.disconnect(old)
        self._legend_place = place
        view.events.resize.connect(place)
        place()
        return node


class Axes(_AxesBase):
    """A 2D axes with pan/zoom camera, tick axes, labels, legend and twinx."""
    _ndim = 2

    def __init__(self, fig, grid):
        super().__init__(fig)
        self._grid = grid
        ts, ls, ks = _fs("axes.titlesize"), _fs("axes.labelsize"), _fs("tick.labelsize")
        self._title = _label(size=ts)
        _size_label_slot(self._title, ts, vertical=False)
        grid.add_widget(self._title, row=0, col=2)
        self._ylabel = _label(rotation=-90, size=ls)
        _size_label_slot(self._ylabel, ls, vertical=True)
        grid.add_widget(self._ylabel, row=1, col=0)
        self._yaxis = _axis_widget("left", ks)
        grid.add_widget(self._yaxis, row=1, col=1)
        self.view = grid.add_view(row=1, col=2, border_color=_FG)
        self._xaxis = _axis_widget("bottom", ks)
        grid.add_widget(self._xaxis, row=2, col=2)
        self._xlabel = _label(size=ls)
        _size_label_slot(self._xlabel, ls, vertical=False)
        grid.add_widget(self._xlabel, row=3, col=2)

        self.view.camera = scene.PanZoomCamera(aspect=None)
        self.view.camera.rect = (0, 0, 1, 1)
        _link_axis(self._xaxis, self.view)
        _link_axis(self._yaxis, self.view)
        self._data_parent = self.view.scene
        self._xlim = self._ylim = None
        self._next_col = 3
        self._gridlines = None

    def _take_col(self):
        c = self._next_col
        self._next_col += 1
        return c

    # ---- limits ----
    def _set_rect(self, x0, x1, y0, y1):
        self.view.camera.rect = (x0, y0, x1 - x0, y1 - y0)
        for t in self._twins:
            t._sync()

    def _autoscale(self):
        r = self.view.camera.rect
        xb = [b[0] for b in (a.bounds() for a in self._all_artists()) if b is not None]
        yb = [b[1] for b in (a.bounds() for a in self._artists) if b is not None]
        x0, x1 = self._xlim or (_pad(min(b[0] for b in xb), max(b[1] for b in xb)) if xb else (r.left, r.right))
        y0, y1 = self._ylim or (_pad(min(b[0] for b in yb), max(b[1] for b in yb)) if yb else (r.bottom, r.top))
        self._set_rect(x0, x1, y0, y1)

    def autoscale(self):
        """Re-fit limits to the current data (like relim(); autoscale_view())."""
        self._xlim = self._ylim = None
        for t in self._twins:
            t._ylim_user = None
        for t in self._twins:
            t._fit_y()
        self._autoscale()

    relim = autoscale

    def set_xlim(self, left=None, right=None):
        if isinstance(left, (tuple, list)):
            left, right = left
        r = self.view.camera.rect
        self._xlim = (r.left if left is None else left, r.right if right is None else right)
        self._set_rect(*self._xlim, r.bottom, r.top)

    def set_ylim(self, bottom=None, top=None):
        if isinstance(bottom, (tuple, list)):
            bottom, top = bottom
        r = self.view.camera.rect
        self._ylim = (r.bottom if bottom is None else bottom, r.top if top is None else top)
        self._set_rect(r.left, r.right, *self._ylim)

    def get_xlim(self):
        r = self.view.camera.rect
        return (r.left, r.right)

    def get_ylim(self):
        r = self.view.camera.rect
        return (r.bottom, r.top)

    def set_aspect(self, aspect):
        self.view.camera.aspect = 1.0 if aspect == "equal" else (None if aspect == "auto" else aspect)

    # ---- labels ----
    def set_xlabel(self, s, fontsize=None):
        _set_label(self._xlabel, s, fontsize, vertical=False)

    def set_ylabel(self, s, fontsize=None):
        _set_label(self._ylabel, s, fontsize, vertical=True)

    def set_title(self, s, fontsize=None):
        _set_label(self._title, s, fontsize, vertical=False)

    def tick_params(self, labelsize=None, axis="both"):
        """Change tick-label font size (points), like matplotlib's tick_params."""
        if labelsize is None:
            return
        widgets = []
        if axis in ("x", "both"):
            widgets.append(self._xaxis)
        if axis in ("y", "both"):
            widgets.append(self._yaxis)
            widgets += [t._yaxis for t in self._twins]
        for aw in widgets:
            _set_axis_tick_size(aw, labelsize)
            _size_axis_slot(aw, labelsize)

    def grid(self, visible=True, color=(0, 0, 0, 0.12)):
        if visible and self._gridlines is None:
            self._gridlines = scene.visuals.GridLines(color=color, parent=self.view.scene)
            self._gridlines.order = -10
        elif self._gridlines is not None:
            self._gridlines.visible = bool(visible)

    def axhline(self, y=0, color="k", linewidth=1.0):
        return scene.visuals.InfiniteLine(pos=y, color=_rgba(color), vertical=False,
                                          parent=self._data_parent)

    def axvline(self, x=0, color="k", linewidth=1.0):
        return scene.visuals.InfiniteLine(pos=x, color=_rgba(color), vertical=True,
                                          parent=self._data_parent)

    def twinx(self):
        t = TwinX(self)
        self._twins.append(t)
        return t

    def cla(self):
        for a in list(self._artists):
            a.remove()
        self._legend_kw = None
        if self._legend_node is not None:
            self._legend_node.parent = None
            self._legend_node = None

    clear = cla


class TwinX(_AxesBase):
    """Second y axis on the right sharing the parent's x axis (see Axes.twinx)."""
    _ndim = 2

    def __init__(self, parent):
        super().__init__(parent.figure)
        self._parent = parent
        self.view = parent.view
        self._data_parent = scene.Node(parent=parent.view.scene)
        self._data_parent.transform = STTransform()
        ls, ks = _fs("axes.labelsize"), _fs("tick.labelsize")
        self._yaxis = _axis_widget("right", ks)
        parent._grid.add_widget(self._yaxis, row=1, col=parent._take_col())
        self._ylabel = _label(rotation=-90, size=ls)
        _size_label_slot(self._ylabel, ls, vertical=True)
        parent._grid.add_widget(self._ylabel, row=1, col=parent._take_col())
        self._ylim_user = None
        self._yfit = (0.0, 1.0)
        parent.view.scene.transform.changed.connect(self._update_axis)
        self._data_parent.transform.changed.connect(self._update_axis)
        self._yaxis.events.resize.connect(self._update_axis)
        self._sync()

    @property
    def _ylims(self):
        return self._ylim_user or self._yfit

    def _fit_y(self):
        yb = [b[1] for b in (a.bounds() for a in self._artists) if b is not None]
        if yb:
            self._yfit = _pad(min(b[0] for b in yb), max(b[1] for b in yb))

    def _autoscale(self):
        self._fit_y()
        self._parent._autoscale()   # handles x, and calls _sync()

    def _sync(self):
        """Map this axes' y range onto the parent's current y range."""
        r = self._parent.view.camera.rect
        y0, y1 = r.bottom, r.top
        t0, t1 = self._ylims
        a = (y1 - y0) / (t1 - t0) if t1 != t0 else 1.0
        b = y0 - a * t0
        self._data_parent.transform.scale = (1.0, a, 1.0)
        self._data_parent.transform.translate = (0.0, b, 0.0)
        self._update_axis()

    def _update_axis(self, event=None):
        try:
            tr = self._yaxis.node_transform(self._data_parent)
            p = tr.map(self._yaxis._axis_ends())
            self._yaxis.axis.domain = (p[0][1], p[1][1])
        except Exception:
            pass   # not laid out yet

    def set_ylim(self, bottom=None, top=None):
        if isinstance(bottom, (tuple, list)):
            bottom, top = bottom
        lo, hi = self._ylims
        self._ylim_user = (lo if bottom is None else bottom, hi if top is None else top)
        self._sync()

    def get_ylim(self):
        return self._ylims

    def set_ylabel(self, s, fontsize=None):
        _set_label(self._ylabel, s, fontsize, vertical=True)

    def tick_params(self, labelsize=None, axis="both"):
        if labelsize is None:
            return
        if axis in ("y", "both"):
            _set_axis_tick_size(self._yaxis, labelsize)
            _size_axis_slot(self._yaxis, labelsize)
        if axis in ("x", "both"):
            self._parent.tick_params(labelsize=labelsize, axis="x")

    # x-axis things go to the parent
    def set_xlim(self, *a, **k):
        self._parent.set_xlim(*a, **k)

    def set_xlabel(self, s, fontsize=None):
        self._parent.set_xlabel(s, fontsize)

    def set_title(self, s, fontsize=None):
        self._parent.set_title(s, fontsize)

    def legend(self, *a, **k):
        return self._parent.legend(*a, **k)

    def _add(self, artist):
        self._artists.append(artist)
        artist._set_order(len(self._parent._all_artists()))
        self._autoscale()
        if self._parent._legend_kw is not None:
            self._parent.legend(**self._parent._legend_kw)
        return artist


class Axes3D(_AxesBase):
    """3D axes: turntable camera, data normalized into a unit box (like matplotlib)."""
    _ndim = 3

    def __init__(self, fig, grid, elev=25.0, azim=-60.0):
        super().__init__(fig)
        self._grid = grid
        ts = _fs("axes.titlesize")
        self._title = _label(size=ts)
        _size_label_slot(self._title, ts, vertical=False)
        grid.add_widget(self._title, row=0, col=0)
        self.view = grid.add_view(row=1, col=0)
        cam = scene.TurntableCamera(fov=30.0, elevation=elev, azimuth=azim)
        self.view.camera = cam
        cam.center = (0.0, 0.0, 0.0)
        cam.scale_factor = 2.2
        self._data_parent = scene.Node(parent=self.view.scene)
        self._data_parent.transform = STTransform()
        self._lims = [None, None, None]
        self._labels = ["", "", ""]
        self._next_col = 1

        o = 0.5
        corners = np.array([[x, y, z] for x in (-o, o) for y in (-o, o) for z in (-o, o)], np.float32)
        edges = [(i, j) for i in range(8) for j in range(i + 1, 8)
                 if np.sum(corners[i] != corners[j]) == 1]
        box = corners[np.array(edges).ravel()]
        self._box = scene.visuals.Line(pos=box, color=(0, 0, 0, 0.18), connect="segments",
                                       method="gl", parent=self.view.scene)
        axes_pos = np.array([[-o, -o, -o], [o, -o, -o],     # x edge
                             [o, -o, -o], [o, o, -o],       # y edge
                             [-o, -o, -o], [-o, -o, o]],    # z edge
                            np.float32)
        self._axes_lines = scene.visuals.Line(pos=axes_pos, color=_FG, connect="segments",
                                              method="gl", parent=self.view.scene)
        self._ticks = scene.visuals.Line(pos=np.zeros((2, 3), np.float32), color=_FG,
                                         connect="segments", method="gl", parent=self.view.scene)
        self._ticktext = scene.visuals.Text(" ", pos=(0, 0, 0), color=_FG,
                                            font_size=_fs("tick.labelsize"),
                                            parent=self.view.scene)
        self._labeltext = scene.visuals.Text(" ", pos=(0, 0, 0), color=_FG,
                                             font_size=_fs("axes.labelsize"),
                                             parent=self.view.scene)
        for v in (self._ticktext, self._labeltext):
            _overlay(v, 500)
        self._rebuild()

    def _take_col(self):
        c = self._next_col
        self._next_col += 1
        return c

    def _limits(self):
        bs = [b for b in (a.bounds() for a in self._artists) if b is not None]
        lims = []
        for i in range(3):
            if self._lims[i] is not None:
                lims.append(self._lims[i])
            elif bs:
                lims.append(_pad(min(b[i][0] for b in bs), max(b[i][1] for b in bs), 0.02))
            else:
                lims.append((0.0, 1.0))
        return lims

    def _rebuild(self):
        lims = self._limits()
        lo = np.array([l[0] for l in lims], float)
        hi = np.array([l[1] for l in lims], float)
        span = np.where(hi - lo == 0, 1.0, hi - lo)
        scale = 1.0 / span
        self._data_parent.transform.scale = tuple(scale)
        self._data_parent.transform.translate = tuple(-lo * scale - 0.5)

        o, off, tl = 0.5, 0.09, 0.03
        texts, tpos, segs = [], [], []
        for i in range(3):
            for t in _nice_ticks(lo[i], hi[i], 5):
                u = (t - lo[i]) * scale[i] - o
                if i == 0:
                    a, b, p = (u, -o, -o), (u, -o - tl, -o), (u, -o - off, -o)
                elif i == 1:
                    a, b, p = (o, u, -o), (o + tl, u, -o), (o + off, u, -o)
                else:
                    a, b, p = (-o, -o, u), (-o - tl, -o - tl, u), (-o - off, -o - off, u)
                segs += [a, b]
                tpos.append(p)
                texts.append(f"{t:g}")
        self._ticks.set_data(pos=np.array(segs, np.float32))
        self._ticktext.text = texts or [" "]
        self._ticktext.pos = np.array(tpos or [(0, 0, 0)], np.float32)
        lpos = [(0, -o - 2.6 * off, -o), (o + 2.6 * off, 0, -o), (-o - 2.2 * off, -o - 2.2 * off, 0)]
        self._labeltext.text = [s or " " for s in self._labels]
        self._labeltext.pos = np.array(lpos, np.float32)

    def _autoscale(self):
        self._rebuild()

    def autoscale(self):
        self._lims = [None, None, None]
        self._rebuild()

    relim = autoscale

    def _set_lim(self, i, lo, hi):
        if isinstance(lo, (tuple, list)):
            lo, hi = lo
        cur = self._limits()[i]
        self._lims[i] = (cur[0] if lo is None else lo, cur[1] if hi is None else hi)
        self._rebuild()

    def set_xlim(self, lo=None, hi=None):
        self._set_lim(0, lo, hi)

    def set_ylim(self, lo=None, hi=None):
        self._set_lim(1, lo, hi)

    def set_zlim(self, lo=None, hi=None):
        self._set_lim(2, lo, hi)

    # In 3D all three axis labels share one text visual, so fontsize applies to all.
    def _set_axlabel(self, i, s, fontsize):
        self._labels[i] = s
        if fontsize is not None:
            self._labeltext.font_size = fontsize
        self._rebuild()

    def set_xlabel(self, s, fontsize=None):
        self._set_axlabel(0, s, fontsize)

    def set_ylabel(self, s, fontsize=None):
        self._set_axlabel(1, s, fontsize)

    def set_zlabel(self, s, fontsize=None):
        self._set_axlabel(2, s, fontsize)

    def set_title(self, s, fontsize=None):
        _set_label(self._title, s, fontsize, vertical=False)

    def tick_params(self, labelsize=None, axis="both"):
        if labelsize is not None:
            self._ticktext.font_size = labelsize

    def view_init(self, elev=None, azim=None):
        if elev is not None:
            self.view.camera.elevation = elev
        if azim is not None:
            self.view.camera.azimuth = azim

    def grid(self, visible=True):
        self._box.visible = bool(visible)

    def cla(self):
        for a in list(self._artists):
            a.remove()
        self._rebuild()

    clear = cla


# --------------------------------------------------------------------------
# sliders
# --------------------------------------------------------------------------

class Slider:
    """matplotlib.widgets.Slider-like control. Use .on_changed(func) and .val."""

    def __init__(self, fig, label, valmin=0.0, valmax=1.0, valinit=None,
                 valstep=None, values=None, valfmt="{:.4g}"):
        self.figure, self.label, self.valfmt = fig, label, valfmt
        if values is None:
            step = valstep or (valmax - valmin) / 100.0
            n = int(round((valmax - valmin) / step)) + 1
            values = valmin + step * np.arange(n)
        self.values = np.asarray(values)
        self._idx = 0 if valinit is None else int(np.argmin(np.abs(self.values - valinit)))
        self._callbacks = {}
        self._cid = 0
        self._qslider = self._qlabel = None

    @property
    def val(self):
        return self.values[self._idx]

    @property
    def index(self):
        return self._idx

    def on_changed(self, func):
        self._cid += 1
        self._callbacks[self._cid] = func
        return self._cid

    def disconnect(self, cid):
        self._callbacks.pop(cid, None)

    def set_val(self, v):
        self._set_index(int(np.argmin(np.abs(self.values - v))))

    def _set_index(self, i):
        i = int(np.clip(i, 0, len(self.values) - 1))
        if i == self._idx:
            return
        self._idx = i
        if self._qslider is not None and self._qslider.value() != i:
            self._qslider.blockSignals(True)
            self._qslider.setValue(i)
            self._qslider.blockSignals(False)
        if self._qlabel is not None:
            self._qlabel.setText(self.valfmt.format(self.val))
        for f in list(self._callbacks.values()):
            f(self.val)


def _qt_modules():
    name = app.use_app().backend_name.lower()
    if "pyqt6" in name:
        from PyQt6 import QtCore, QtWidgets
    elif "pyside6" in name:
        from PySide6 import QtCore, QtWidgets
    elif "pyqt5" in name:
        from PyQt5 import QtCore, QtWidgets
    elif "pyside2" in name:
        from PySide2 import QtCore, QtWidgets
    else:
        return None
    return QtCore, QtWidgets


# --------------------------------------------------------------------------
# figure
# --------------------------------------------------------------------------

class Figure:
    def __init__(self, figsize=(6.4, 4.8), dpi=100, title="vplot",
                 bgcolor="white", samples=None):
        size = (int(figsize[0] * dpi), int(figsize[1] * dpi))
        kw = {"config": {"samples": samples}} if samples else {}
        self.canvas = scene.SceneCanvas(keys="interactive", size=size, bgcolor=bgcolor,
                                        title=title, show=False, dpi=SCREEN_DPI, **kw)
        self.grid = self.canvas.central_widget.add_grid(margin=6)
        self.axes = []
        self._gca = None
        self._title = title
        self._sliders = []
        self._active_slider = 0
        self._container = None
        self._closed = False
        self.canvas.events.key_press.connect(self._on_key)
        self.canvas.events.close.connect(lambda event: self._forget())
        _state["figs"].append(self)
        _state["fig"] = self

    def _forget(self):
        """Drop this figure from pyplot state (called when its window closes)."""
        self._closed = True
        if self in _state["figs"]:
            _state["figs"].remove(self)
        if _state["fig"] is self:
            _state["fig"] = _state["figs"][-1] if _state["figs"] else None

    # ---- layout ----
    def add_subplot(self, *args, projection=None, **kw):
        if len(args) == 1 and isinstance(args[0], int) and args[0] >= 111:
            args = tuple(int(d) for d in str(args[0]))
        if not args:
            args = (1, 1, 1)
        nrows, ncols, index = args
        row, col = divmod(index - 1, ncols)
        cell = self.grid.add_grid(row=row, col=col)
        if projection == "3d":
            ax = Axes3D(self, cell, **kw)
        else:
            ax = Axes(self, cell)
        self.axes.append(ax)
        self._gca = ax
        return ax

    def subplots(self, nrows=1, ncols=1, sharex=False, sharey=False,
                 projection=None, squeeze=True):
        axs = np.empty((nrows, ncols), dtype=object)
        for r in range(nrows):
            for c in range(ncols):
                axs[r, c] = self.add_subplot(nrows, ncols, r * ncols + c + 1,
                                             projection=projection)
        if projection != "3d":
            flat = list(axs.ravel())
            for a in flat[1:]:
                if sharex:
                    _link(flat[0], a, "x")
                if sharey:
                    _link(flat[0], a, "y")
        if squeeze:
            if axs.size == 1:
                return axs[0, 0]
            if nrows == 1 or ncols == 1:
                return axs.ravel()
        return axs

    def gca(self):
        if self._gca is None:
            self.add_subplot(1, 1, 1)
        return self._gca

    # ---- decorations ----
    def colorbar(self, mappable, ax=None, label="", **kw):
        ax = ax or mappable._ax
        ax = getattr(ax, "_parent", None) or ax     # twin -> parent grid
        opts = dict(orientation="right", label=label, label_color=_FG,
                    clim=mappable.clim, border_color=_FG, border_width=1.0)
        opts.update(kw)
        cb = scene.widgets.ColorBarWidget(cmap=mappable._cmap, **opts)
        # ColorBarWidget has no font-size option; adjust its inner text visuals
        # where this VisPy version exposes them.
        ks, ls = _fs("tick.labelsize"), _fs("axes.labelsize")
        inner = getattr(cb, "_colorbar", None)
        if inner is not None:
            _set_text_size(inner, ls, "_label")
            for t in getattr(inner, "_ticks", []) or []:
                if hasattr(type(t), "font_size"):
                    t.font_size = ks
        cb.width_max = int(ks * PT * 4.5 + ls * PT * 1.8 + 30)
        ax._grid.add_widget(cb, row=1, col=ax._take_col())
        mappable._colorbars.append(cb)
        return cb

    def suptitle(self, s):
        self.canvas.title = s

    # ---- sliders ----
    def add_slider(self, label, valmin=0.0, valmax=1.0, valinit=None,
                   valstep=None, values=None, valfmt="{:.4g}"):
        """Add a slider below the plot. Returns a Slider; use .on_changed(func).

        Pass either valmin/valmax/valstep or an explicit array of `values`.
        With non-Qt backends, use Left/Right arrow keys (Up/Down picks slider).
        """
        s = Slider(self, label, valmin, valmax, valinit, valstep, values, valfmt)
        self._sliders.append(s)
        return s

    def _on_key(self, event):
        if not self._sliders or event.key is None:
            return
        name = event.key.name
        s = self._sliders[self._active_slider]
        step = 10 if "Shift" in [m.name for m in event.modifiers] else 1
        if name == "Right":
            s._set_index(s._idx + step)
        elif name == "Left":
            s._set_index(s._idx - step)
        elif name in ("Up", "Down"):
            d = -1 if name == "Up" else 1
            self._active_slider = (self._active_slider + d) % len(self._sliders)

    def _build_qt(self):
        mods = _qt_modules()
        if mods is None:
            _warn_once("qt", "vplot: slider panel needs a Qt backend; "
                             "use Left/Right arrow keys instead")
            return False
        QtCore, QtWidgets = mods
        app.use_app().create()
        fig = self

        class _Window(QtWidgets.QWidget):
            def closeEvent(self, event):
                fig._forget()
                event.accept()

        win = _Window()
        win.setWindowTitle(self._title)
        lay = QtWidgets.QVBoxLayout(win)
        lay.setContentsMargins(0, 0, 0, 6)
        lay.addWidget(self.canvas.native, 1)
        horiz = QtCore.Qt.Orientation.Horizontal
        for s in self._sliders:
            row = QtWidgets.QHBoxLayout()
            row.setContentsMargins(10, 0, 10, 0)
            name = QtWidgets.QLabel(s.label)
            sl = QtWidgets.QSlider(horiz)
            sl.setRange(0, len(s.values) - 1)
            sl.setValue(s._idx)
            val = QtWidgets.QLabel(s.valfmt.format(s.val))
            val.setMinimumWidth(70)
            sl.valueChanged.connect(lambda i, s=s: s._set_index(i))
            s._qslider, s._qlabel = sl, val
            row.addWidget(name)
            row.addWidget(sl, 1)
            row.addWidget(val)
            lay.addLayout(row)
        w, h = self.canvas.size
        win.resize(w, h + 34 * len(self._sliders))
        self._container = win
        return True

    # ---- output ----
    def show(self):
        if self._sliders and self._container is None and self._build_qt():
            pass
        if self._container is not None:
            self._container.show()
        else:
            self.canvas.show()

    def savefig(self, fname):
        img = self.canvas.render()
        if str(fname).lower().endswith(".png"):
            from vispy.io import write_png
            write_png(fname, img)
        else:
            import imageio
            imageio.imwrite(fname, img[..., :3])

    def close(self):
        self._forget()
        for obj in (self._container, self.canvas):
            if obj is not None:
                try:
                    obj.close()
                except Exception:
                    pass   # already closed by the user


def _link(a, b, axis):
    """Keep a and b's x (or y) ranges in sync while panning/zooming."""
    busy = [False]

    def make(src, dst):
        def cb(event=None):
            if busy[0]:
                return
            busy[0] = True
            try:
                r, d = src.view.camera.rect, dst.view.camera.rect
                if axis == "x" and (r.left, r.right) != (d.left, d.right):
                    dst.view.camera.rect = (r.left, d.bottom, r.width, d.height)
                elif axis == "y" and (r.bottom, r.top) != (d.bottom, d.top):
                    dst.view.camera.rect = (d.left, r.bottom, d.width, r.height)
            finally:
                busy[0] = False
        return cb

    a.view.scene.transform.changed.connect(make(a, b))
    b.view.scene.transform.changed.connect(make(b, a))


# --------------------------------------------------------------------------
# pyplot-style state machine
# --------------------------------------------------------------------------

_state = {"figs": [], "fig": None}


def figure(figsize=(6.4, 4.8), dpi=100, title="vplot", **kw):
    return Figure(figsize=figsize, dpi=dpi, title=title, **kw)


def subplots(nrows=1, ncols=1, figsize=None, sharex=False, sharey=False,
             projection=None, squeeze=True, **kw):
    if figsize is None:
        figsize = (6.4, 4.8) if nrows * ncols == 1 else (4.0 * ncols, 3.2 * nrows)
    fig = Figure(figsize=figsize, **kw)
    return fig, fig.subplots(nrows, ncols, sharex=sharex, sharey=sharey,
                             projection=projection, squeeze=squeeze)


def gcf():
    return _state["fig"] or Figure()


def gca():
    return gcf().gca()


def subplot(*args, projection=None):
    return gcf().add_subplot(*args, projection=projection)


def plot(*a, **k):
    return gca().plot(*a, **k)


def scatter(*a, **k):
    return gca().scatter(*a, **k)


def text(*a, **k):
    return gca().text(*a, **k)


def axhline(*a, **k):
    return gca().axhline(*a, **k)


def axvline(*a, **k):
    return gca().axvline(*a, **k)


def xlabel(s, fontsize=None):
    gca().set_xlabel(s, fontsize)


def ylabel(s, fontsize=None):
    gca().set_ylabel(s, fontsize)


def zlabel(s, fontsize=None):
    gca().set_zlabel(s, fontsize)


def title(s, fontsize=None):
    gca().set_title(s, fontsize)


def tick_params(labelsize=None, axis="both"):
    gca().tick_params(labelsize=labelsize, axis=axis)


def xlim(*a):
    ax = gca()
    return ax.set_xlim(*a) if a else ax.get_xlim()


def ylim(*a):
    ax = gca()
    return ax.set_ylim(*a) if a else ax.get_ylim()


def zlim(*a):
    gca().set_zlim(*a)


def legend(**k):
    return gca().legend(**k)


def grid(visible=True):
    gca().grid(visible)


def twinx(ax=None):
    t = (ax or gca()).twinx()
    gcf()._gca = t
    return t


def colorbar(mappable=None, ax=None, label="", **kw):
    if mappable is None:
        ax0 = gca()
        cands = [a for a in getattr(ax0, "_all_artists", lambda: ax0._artists)()
                 if isinstance(a, Scatter)]
        if not cands:
            raise RuntimeError("No scatter to make a colorbar for")
        mappable = cands[-1]
    return mappable._ax.figure.colorbar(mappable, ax=ax, label=label, **kw)


def savefig(fname):
    gcf().savefig(fname)


def close(fig=None):
    if fig == "all":
        for f in list(_state["figs"]):
            f.close()
    else:
        (fig or gcf()).close()


def show(block=True):
    """Show open figures and (by default) run the event loop until they close.

    Like matplotlib, a figure is forgotten once its window is closed, so the
    next show() only displays figures created since then. With block=False the
    windows stay registered until you close them (needs a running GUI loop,
    e.g. %gui qt in IPython).
    """
    figs = list(_state["figs"])
    for f in figs:
        f.show()
    if block and figs:
        app.run()
        # The loop exits once every window is closed; make sure none linger.
        for f in figs:
            f.close()