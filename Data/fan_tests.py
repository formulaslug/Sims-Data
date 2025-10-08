from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import polars as pl
import numpy as np

df_fans_0 = pl.read_parquet("./FS-3/08172025/08172025_28_45C_40C_~29Cambient_0fans.parquet").slice(69020, 60500)
df_fans_100 = pl.read_parquet("./FS-3/08172025/08172025_27_45C_35C_~28Cambient_100fans.parquet").slice(20, 60500)
temp_col_names = [f"ACC_SEG{n}_TEMPS_CELL{m}" for n in range(5) for m in range(6)]
df_fans_0_temps = df_fans_0.select(temp_col_names)
df_fans_100_temps = df_fans_100.select(temp_col_names)

fans_0_mean = df_fans_0_temps.cast(pl.Float64).mean_horizontal().alias('fans_0_mean_temp')
fans_0_max = df_fans_0_temps.max_horizontal().alias('fans_0_max_temp')
fans_0_min = df_fans_0_temps.min_horizontal().alias('fans_0_min_temp')

fans_100_mean = df_fans_100_temps.cast(pl.Float64).mean_horizontal().alias('fans_100_mean_temp')
fans_100_max = df_fans_100_temps.max_horizontal().alias('fans_100_max_temp')
fans_100_min = df_fans_100_temps.min_horizontal().alias('fans_100_min_temp')

fig, axes = plt.subplots(2, 1)
ax0: Axes = axes[0]
ax1: Axes = axes[1]

def ax_plot_df(ax: Axes, df: pl.DataFrame):
    for col in df.columns:
        ax.plot(np.arange(df.height)*0.01, df[col], label=col)
def ax_plot_series(ax: Axes, s: pl.Series):
    ax.plot(np.arange(s.count())*0.01, s, label=s.name)

ax_plot_series(ax0, fans_0_mean)
ax_plot_series(ax0, fans_0_max)
ax_plot_series(ax0, fans_0_min)
ax_plot_series(ax1, fans_100_mean)
ax_plot_series(ax1, fans_100_max)
ax_plot_series(ax1, fans_100_min)
# ax_plot_df(ax0, df_fans_0_temps)
# ax_plot_df(ax1, df_fans_100_temps)

ax0.set_title("fans_0")
ax1.set_title("fans_100")

ax0.set_ybound(30, 55)
ax1.set_ybound(30, 55)
ax0.set_xbound(-10, 610)
ax1.set_xbound(-10, 610)

# Legends outside plots
ax0.legend(bbox_to_anchor=(1,1))
ax1.legend(bbox_to_anchor=(1,1))

ax0.set_ylabel("temperature (C)")
ax1.set_ylabel("temperature (C)")
ax0.set_xlabel("time (s)")
ax1.set_xlabel("time (s)")
plt.show()
