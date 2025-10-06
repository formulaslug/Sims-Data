import numpy as np
import matplotlib.pyplot as plt
import polars as pl
from numpy.linalg import lstsq

# Load & filter
newData = pl.read_csv("unifiedTraining.csv", separator=",")
newData = newData.filter((pl.col('SA') >= -0.10) & (pl.col('SA') <= 0.10))
newData = newData.with_columns(pl.col('NFY').abs())
newData = newData.filter(pl.col('NFY') < 1.5)

x_column = 'FZ'
y_column = 'NFY'
x = newData[x_column].to_numpy()
y = newData[y_column].to_numpy()

# Design matrix for quadratic y = a x^2 + b x + c
X = np.column_stack([x**2, x, np.ones_like(x)])
coeffs, *_ = lstsq(X, y, rcond=None)
a, b, c = coeffs

# Force downward-opening parabola: ensure a < 0
if a >= 0:
    a = -abs(a)
    A_fixed = np.column_stack([x, np.ones_like(x)])
    y_adj = y - a * x**2
    bc, *_ = lstsq(A_fixed, y_adj, rcond=None)
    b, c = bc
print(a,b,c)
# Print the quadratic in readable form
print(f"Fitted quadratic: y = {a:.6g} x^2 + {b:.6g} x + {c:.6g}")
# Also print vertex and a if useful
vertex_x = -b / (2*a)
vertex_y = a*vertex_x**2 + b*vertex_x + c
print(f"  (a = {a:.6g} < 0, vertex at x = {vertex_x:.6g}, y = {vertex_y:.6g})")

# Plot hexbin heatmap + fitted parabola
plt.figure(figsize=(10,6))
hb = plt.hexbin(x, y, gridsize=80, cmap='inferno', mincnt=1)
plt.colorbar(hb, label='counts')

xs = np.linspace(np.min(x), np.max(x), 400)
ys = a*xs**2 + b*xs + c
plt.plot(xs, ys, color='cyan', linewidth=2, label=f'y = {a:.3g}x² + {b:.3g}x + {c:.3g}')

plt.xlabel(x_column)
plt.ylabel(y_column)
#plt.legend()
plt.grid(alpha=0.3)
plt.show()
