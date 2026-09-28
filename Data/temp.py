import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.cm import get_cmap
from matplotlib.colors import Normalize
from os import listdir

from Data.FSLib.AnalysisFunctions import read, simpleTimeCol

# new_paths = ["FS-4/TestTuneAndTech_Sep182026/" + x for x in listdir("../fs-data/FS-4/TestTuneAndTech_Sep182026/")]
# dfs = [read(x) for x in new_paths]

# for df, path in zip(dfs, listdir("../fs-data/FS-4/TestTuneAndTech_Sep182026/")):
#     t = df["Time"]
#     plt.plot(t, df["SME_TRQSPD_Speed"]/10, label = "RPM/10")
#     plt.plot(t, df["SME_TRQSPD_Torque"], label = "SME_TRQSPD_Torque")
#     plt.plot(t, df["VCU_ACCEL_PEDAL_TRAVEL"], label = "Pedal Travel")
#     plt.plot(t, df["SME_TEMP_BusCurrent"], label="current")
#     plt.plot(t, df["VCU_STEERING_ANGLE"], label="steering angle")
#     plt.legend()
#     plt.title(path)
#     plt.show()


path = "FS-4/TestTuneAndTech_Sep192026/163551_probablySkidpad.parquet"
# 155420, 155424, 155428
path2 = "FS-3/08172025/08172025_26autox1.parquet"
path2 = "FS-3/08102025/08102025Endurance1_FirstHalf.parquet"

df = read(path)
df2 = read(path2, fs2or3=True)
# df = df.insert_column(0, simpleTimeCol(df))

t = df["Time"]
# [x for x in df.columns if "PEDAL" in x]

voltages = [[f"BATT_MOD{x}_VOLTS_CELL{y}" for y in range(6)] for x in range (5)]
temps = [[f"BATT_MOD{x}_TEMPS_CELL{y}" for y in range(12)] for x in range (5)]
flat_temps = [f"BATT_MOD{x}_TEMPS_CELL{y}" for y in range(12) for x in range (5)]

plt.plot(t, df["SME_TRQSPD_Speed"]/10, label = "RPM/10")
plt.plot(t, df["SME_TRQSPD_Torque"], label = "SME_TRQSPD_Torque")
plt.plot(t, df["VCU_ACCEL_PEDAL_TRAVEL"], label = "Pedal Travel")
plt.plot(t, df["SME_TEMP_BusCurrent"], label="current")
plt.legend()
plt.show()

temp_sensors = ["BATT_TRAY_TEMPS_BOLTED_CONNECTION", "BATT_TRAY_TEMPS_BUSBAR", "BATT_TRAY_TEMPS_PACK_FUSE", "BATT_TRAY_TEMPS_COWLING", "BATT_TRAY_TEMPS_INTAKE"]

flsus = "TPERIPH_FL_DATA_SUSTRAVEL"
frsus = "TPERIPH_FR_DATA_SUSTRAVEL"
blsus = "TPERIPH_BL_DATA_SUSTRAVEL"
brsus = "TPERIPH_BR_DATA_SUSTRAVEL"

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(t, df[flsus].rolling_mean(11), label = "FL")
ax.plot(t, df[frsus].rolling_mean(11), label = "FR")
ax.plot(t, df[blsus].rolling_mean(11), label = "BL")
ax.plot(t, df[brsus].rolling_mean(11), label = "BR")
ax.legend()
fig.show()

plt.plot(t, df["VCU_STEERING_ANGLE"])
plt.show()


fig = plt.figure()
ax = fig.add_subplot(111)
for sensor in temp_sensors:
    ax.plot(df[sensor], label=sensor)
ax.legend()
fig.show()

busC = df["SME_TEMP_BusCurrent"]
busV = df["SME_TEMP_DC_Bus_V"]

batV = df["BATT_POWER_PACK_VOLTAGE"]
batC = df["BATT_POWER_CURRENT"]

plt.plot(t, (batV - busV), label="battery voltage - bus voltage")
plt.plot(t, (batV - busV)*busC, label="(batV - busV) * busC")
plt.plot(t, (batV - busV)*batC, label="(batV - busV) * busC")
plt.plot(t, batC, label = "battery current")
plt.plot(t, busC, label = "bus current")
plt.legend()
plt.show()

t2 = simpleTimeCol(df2)
volts2 = [f"ACC_SEG{x}_VOLTS_CELL{y}" for x in range(5) for y in range (6)]
dfVolts2 = df2[volts2].sum_horizontal()
vDiff2 = df2["SME_TEMP_DC_Bus_V"] - dfVolts2
df2 = df2.with_columns((vDiff2 * df2["SME_TEMP_BusCurrent"]).alias("PowerLoss"), (dfVolts2).alias("PackVoltage"))
df2 = df2.filter(pl.col("SME_TEMP_BusCurrent") > 100).filter(pl.col("SME_TEMP_BusCurrent") < 1000)

plt.plot(t2, df2["SME_TEMP_BusCurrent"], label="current")
plt.plot(t2, df2["SME_TRQSPD_Speed"]/10, label="rpm/10")
plt.plot(t2, df2["ETC_STATUS_PEDAL_TRAVEL"], label="pedal travel")
plt.legend()
plt.show()

fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
ax2.plot(t, (batV - busV)*busC, label="(batV - busV) * busC")
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Power Loss (W)")
fig2.show()

fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
ax2.plot(t, np.convolve((batV - busV)/busC, np.ones(201)/201, mode='same'), label="(batV - busV) * busC")
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Resistance (Ohms)")
fig2.show()

fig3 = plt.figure()
ax3 = fig3.add_subplot(111)
ax3.plot(t2, vDiff2 * df2["SME_TEMP_BusCurrent"])
fig3.show()

np.polyfit(busC, (batV - busV), 1)
np.polyfit(df2["SME_TEMP_BusCurrent"], df2["PackVoltage"] - df2["SME_TEMP_DC_Bus_V"], 1)

plt.scatter(df2["SME_TEMP_BusCurrent"], df2["PackVoltage"] - df2["SME_TEMP_DC_Bus_V"], label = "Pack Voltage")
X = np.arange(100, 600, 10)
plt.plot(X, 0.0390288*X - 1.567)
plt.show()

plt.scatter(busC, (batV - busV), label = "Bus Voltage")
plt.show()

for temp in temps:
    plt.plot(t, df[temp], label = temp)
plt.legend()
plt.show()

# graph each set of temperatures as a separate plane in a 3d plot with an isometric view so all temperatures can be seen at the same time. 
# Every 2 are on a new row (0-11)
# The temperatures are by module. Do this for the last row of the parquet file

# 3d plot

cmap = get_cmap('viridis')

x = np.linspace(0, 2, 7)
y = np.arange(0, 3, 1)
X, Y = np.meshgrid(x, y)
Z = np.zeros_like(X)
norm = Normalize(vmin=df[flat_temps].min_horizontal().min(), vmax=df[flat_temps].max_horizontal().max())

minTemp = df[flat_temps].min_horizontal().min()
maxTemp = df[flat_temps].max_horizontal().max()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for i, temp in enumerate(temps):
    color_data = df[temp][-1].to_numpy().reshape(2,6)
    
    facecolors = cmap(norm(color_data))
    ax.plot_surface(X, Y, Z+i,  cmap='viridis', alpha=0.5)
# fig.colorbar(ax.plot_surface(X, Y, Z, facecolors=df[temps[0]][-1].to_numpy().reshape(2,6), cmap='viridis', alpha=0.5), ax=ax, shrink=0.5, aspect=5)
fig.show()


n_steps = len(df[temps[0]])

# Create figure with space for slider
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
plt.subplots_adjust(bottom=0.2)

# Add slider axis
slider_ax = plt.axes([0.2, 0.05, 0.6, 0.03])
slider = Slider(slider_ax, 'Time Step', 0, n_steps - 1, valinit=n_steps - 1, valstep=1, color='steelblue')

# Store the surface plots so we can update them
surfaces = []
cbar = None

def update(val):
    global surfaces, cbar
    
    # Get current slider value
    idx = int(slider.val)
    
    # Clear the axis
    ax.clear()
    
    # Redraw all surfaces at this time step
    for i, temp in enumerate(temps):
        surf = ax.plot_surface(X, Y, Z+i*1, facecolors=cmap(norm(df[temp][idx].to_numpy().reshape(2, 6))),
                              cmap='viridis', alpha=0.5)
    
    # Redraw colorbar
    if cbar:
        cbar.clear()
    cbar = fig.colorbar(ax.plot_surface(X, Y, Z, facecolors=cmap(norm(df[temps[0]][idx].to_numpy().reshape(2, 6))), 
                                        cmap='viridis', alpha=0.5), 
                       ax=ax, shrink=0.5, aspect=5)
    
    # Reset labels and limits
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    fig.canvas.draw_idle()

# Connect slider to update function
slider.on_changed(update)

# Initial plot
update(n_steps - 1)

fig.show()