"""
Examples for vplot. Run one at a time:

    python vplot_examples.py lines      # subplots, legend, twinx, 1M-point line
    python vplot_examples.py scatter    # 2D scatter with colorbar
    python vplot_examples.py scatter3d  # 3D scatter + slider through a 4th dimension
    python vplot_examples.py pyplot     # pyplot-style state machine calls
"""
import sys

import numpy as np

from Data.FSLib import vplot as vp


def lines():
    x = np.linspace(0, 50, 1_000_000)
    fig, (ax1, ax2) = vp.subplots(2, 1, figsize=(8, 6), sharex=True)

    ax1.plot(x, np.sin(x), label="sin(x)")
    ax1.plot(x, 0.5 * np.cos(3 * x), "r", label="0.5 cos(3x)")
    ax1.set_title("One million points per line")
    ax1.set_ylabel("amplitude")
    ax1.legend()
    ax1.grid()

    ax2.plot(x[::5000], np.exp(x[::5000] / 20), "k-o", ms=3, label="exp(x/20)")
    ax2.set_ylabel("growth")
    ax2.set_xlabel("time [s]")
    t = ax2.twinx()
    t.plot(x[::5000], np.log1p(x[::5000]), "C2", label="log(1+x)")
    t.set_ylabel("log scale")
    ax2.legend(loc="upper left")
    vp.show()


def scatter():
    rng = np.random.default_rng(0)
    n = 200_000
    x, y = rng.normal(size=(2, n))
    r = np.hypot(x, y)
    fig, ax = vp.subplots()
    sc = ax.scatter(x, y, c=r, s=4, cmap="viridis", alpha=0.6)
    fig.colorbar(sc, label="radius")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"{n:,} points")
    ax.set_aspect("equal")
    vp.show()

def pyplot():
    x = np.linspace(0, 10, 500)
    vp.figure()
    vp.plot(x, np.sin(x), label="sin")
    vp.plot(x, np.cos(x), "--", label="cos")   # dashes are drawn solid
    vp.scatter(x[::25], np.sin(x[::25]), c="k", s=16, label="samples")
    vp.xlabel("x")
    vp.ylabel("y")
    vp.title("pyplot style")
    vp.legend()
    vp.ylim(-1.5, 1.5)
    vp.show()

rng = np.random.default_rng(1)
n = 50_000
x, y, z = rng.uniform(-1, 1, size=(3, n))
ts = np.linspace(0, 2 * np.pi, 120)

def field(t):
    return np.sin(3 * x + t) * np.cos(3 * y - t) + z * np.cos(t)

fig, ax = vp.subplots(projection="3d", figsize=(7, 6))
sc = ax.scatter(x, y, z, c=field(ts[0]), s=3, cmap="coolwarm", vmin=-2, vmax=2)
fig.colorbar(sc, label="f(x, y, z, t)")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.set_title("drag to rotate, slider (or arrow keys) to scan t", fontsize=6)

s = fig.add_slider("t", values=ts)
s.on_changed(lambda t: sc.set_data(c=field(t)))
vp.show()
