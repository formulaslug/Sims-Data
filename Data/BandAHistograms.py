## Magnetic Field and Accelerometer Histograms

# df.columns = ['time', 'voltage', 'state_of_charge', 'charge_rate', 'xA', 'yA', 'zA', 'xG', 'yG', 'zG','ISM330.temp', 'xB', 'yB', 'zB', 'MMC5983.temp']
from Data.FSLib.IMUCalibrationFunctions import *

# df = pl.read_parquet("Nathaniel_IMU_Data/Orientation.parquet")

df = pl.read_parquet("Nathaniel_IMU_Data/realTrip1.parquet")

df = df.insert_column(0, (df["xA"].pow(2) + df["yA"].pow(2) + df["zA"].pow(2)).sqrt().alias("mA")) #Magnitude of the 3 axis vector
df = df.insert_column(0, pl.Series((np.ones(df.shape[0])*1000)).alias("outA")) #Insert expected output column. Will be cut down to size automatically

df = df.insert_column(0, (df["xB"].pow(2) + df["yB"].pow(2) + df["zB"].pow(2)).sqrt().alias("mB")) #Magnitude of the 3 axis vector

counts, bins = np.histogram(df["mB"].to_numpy(), bins=1000)
plt.stairs(counts, bins)
plt.xlabel('Magnetic Field Magnitude (Gauss)')
plt.ylabel('Counts')
plt.title('Histogram of Magnetic Field Magnitude')
plt.axvline(df["mB"].median(), color='red', linestyle='dashed', linewidth=1, label='Median') #type: ignore
plt.legend()
plt.tight_layout()
plt.show()

counts, bins = np.histogram(df["mA"].to_numpy(), bins=1000)
plt.stairs(counts, bins)
plt.xlabel('Acceleration Magnitude (m/s²)')
plt.ylabel('Counts')
plt.title('Histogram of Acceleration Magnitude')
plt.axvline(df["mA"].median(), color='red', linestyle='dashed', linewidth=1, label='Median') #type: ignore
plt.legend()
plt.tight_layout()
plt.show()