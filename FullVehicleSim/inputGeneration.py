import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from Data.FSLib.AnalysisFunctions import simpleTimeCol


df1 = pl.read_parquet("C:/Projects/FormulaSlug/fs-data/FS-3/PreparedData/CombinedEndurance_0810_0817_2025.parquet")

timecol = simpleTimeCol(df1)
throttle = df1["ETC_STATUS_PEDAL_TRAVEL"]/100.0
frontBrakes = np.clip((df1["ETC_STATUS_BRAKE_SENSE_VOLTAGE"] - 0.33)/2.64, 0, 1)*2000 # Scale to PSI
rearBrakes = frontBrakes # No rear brake sense voltage, so just copy front for now.
steeringAngle = np.zeros(df1.height) # No steering angle data, so just set to 0 for now.

plt.plot(timecol, df1["TMAIN_DATA_STEER"])
plt.show()