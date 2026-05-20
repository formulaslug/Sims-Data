import polars as pl
import matplotlib.pyplot as plt
from scipy import interpolate
import numpy as np

flparams = (5,3,2,1.5)
frparams = (5,3.33,1,3.33)
blparams = (7,3,3,3)
brparams = (7,2.66,3.33,3)

#Make the interpolationerator
highspeed_curves = pl.read_csv('data/12-12-highspeed.csv')
lowspeed_curves = pl.read_csv('data/12-12-lowspeed.csv')

nasty_curves = pl.concat([highspeed_curves, lowspeed_curves], how="horizontal")

curves = pl.DataFrame()

settings = {"0-4.3 0-4.3": (0,4.3,0,4.3),
            "0-3 0-3": (0,3,0,3),
            "0-2 0-2": (0,2,0,2),
            "0-1 0-1": (0,1,0,1),
            "0-0 0-0": (0,0,0,0),
            "2-4.3 2-4.3": (2,4.3,2,4.3),
            "4-4.3 4-4.3": (4,4.3,4,4.3),
            "6-4.3 6-4.3": (6,4.3,6,4.3),
            "10-4.3 10-4.3": (10,4.3,10,4.3),
            "15-4.3 15-4.3": (15,4.3,15,4.3),
            "25-4.3 25-4.3": (25,4.3,25,4.3)}

for key in settings.keys():
    tdf = nasty_curves.with_columns(
            (pl.col(key + " X").alias('velocity')),
            (pl.col(key + " Y").alias('force')))
            #(pl.lit(float(settings[key][0])).alias('lsc')),
            #(pl.lit(float(settings[key][1])).alias('hsc')),
            #(pl.lit(float(settings[key][2])).alias('lsr')),
            #(pl.lit(float(settings[key][3])).alias('hsr')))

    #print(tdf["velocity", "force", "lsc", "hsc", "lsr", "hsr"])
    tdf = tdf.drop_nulls() 
    tdf = tdf.sort('velocity')

    tdf = tdf.with_columns(
        pl.when(pl.col("force") < 0)
          .then(-pl.col("velocity").abs())  # make velocity negative
          .otherwise(pl.col("velocity").abs())  # keep positive otherwise
          .alias("velocity")
    )

    v_new = np.arange(0, 10.0 + 1e-9, 0.05)
    f_new = np.interp(v_new, tdf['velocity'], tdf['force'])

    tdf = pl.DataFrame({'velocity': v_new, 'force': f_new})

    tdf = tdf.with_columns(
            (pl.lit(float(settings[key][0])).alias('lsc')),
            (pl.lit(float(settings[key][1])).alias('hsc')),
            (pl.lit(float(settings[key][2])).alias('lsr')),
            (pl.lit(float(settings[key][3])).alias('hsr')))


    curves = pl.concat([curves, tdf["velocity", "force", "lsc", "hsc", "lsr", "hsr"]], how="vertical")
#curves = curves.sort(["velocity"])
interperator = interpolate.NearestNDInterpolator(curves["velocity", "lsc", "hsc", "lsr", "hsr"], curves["force"])


# Source file generated with c++ decoder program using 100ms cache and forward fill, from fs3norcal.log
all_data = pl.read_parquet('data/fs3norcal_100ms.parquet')


rc = all_data[['Time_ms', 
              'TPERIPH_BL_DATA_SUSTRAVEL', 
              'TPERIPH_BR_DATA_SUSTRAVEL', 
              'TPERIPH_FR_DATA_SUSTRAVEL', 
              'TPERIPH_FL_DATA_SUSTRAVEL']]

rd = (rc.filter(pl.col('Time_ms')>40000)).filter(pl.col('Time_ms')<95000)

rd = rd.with_columns(
    pl.col("TPERIPH_BL_DATA_SUSTRAVEL").rolling_mean(window_size=5).alias("TPERIPH_BL_DATA_SUSTRAVEL"),
    pl.col("TPERIPH_BR_DATA_SUSTRAVEL").rolling_mean(window_size=5).alias("TPERIPH_BR_DATA_SUSTRAVEL"),
    pl.col("TPERIPH_FL_DATA_SUSTRAVEL").rolling_mean(window_size=5).alias("TPERIPH_FL_DATA_SUSTRAVEL"),
    pl.col("TPERIPH_FR_DATA_SUSTRAVEL").rolling_mean(window_size=5).alias("TPERIPH_FR_DATA_SUSTRAVEL")
)

# Function to produce force values
def suspensionForce(position, velocity, params):
    lsc = params[0]
    hsc = params[1]
    lsr = params[2]
    hsr = params[3]

    
    try:
      pos_in = position/25.4
      force_lb = pos_in * 200
      velocity = velocity / 25.4 # convert from mm/s to in/s
      force_lb += interperator(np.array([velocity, lsc, hsc, lsr, hsr]))
      print(interperator(np.array([velocity, lsc, hsc, lsr, hsr])))
    except:
      print("oopsie")
      return None
    return force_lb

rd = rd.with_columns(
        (pl.col('TPERIPH_BL_DATA_SUSTRAVEL').diff() / (pl.col('Time_ms').diff()/1000)).alias('BL_SUSVELOCITY'), #mm/s
        (pl.col('TPERIPH_BR_DATA_SUSTRAVEL').diff() / (pl.col('Time_ms').diff()/1000)).alias('BR_SUSVELOCITY'),
        (pl.col('TPERIPH_FL_DATA_SUSTRAVEL').diff() / (pl.col('Time_ms').diff()/1000)).alias('FL_SUSVELOCITY'),
        (pl.col('TPERIPH_FR_DATA_SUSTRAVEL').diff() / (pl.col('Time_ms').diff()/1000)).alias('FR_SUSVELOCITY'))

rd = rd.with_columns([
    pl.struct(["TPERIPH_BL_DATA_SUSTRAVEL", "BL_SUSVELOCITY"])
      .map_elements(
          lambda row: suspensionForce(row["TPERIPH_BL_DATA_SUSTRAVEL"], row["BL_SUSVELOCITY"], blparams),
          return_dtype=pl.Float64
      )
      .alias("BL_SUSFORCE"),

    pl.struct(["TPERIPH_BR_DATA_SUSTRAVEL", "BR_SUSVELOCITY"])
      .map_elements(
          lambda row: suspensionForce(row["TPERIPH_BR_DATA_SUSTRAVEL"], row["BR_SUSVELOCITY"], brparams),
          return_dtype=pl.Float64
      )
      .alias("BR_SUSFORCE"),

    pl.struct(["TPERIPH_FL_DATA_SUSTRAVEL", "FL_SUSVELOCITY"])
      .map_elements(
          lambda row: suspensionForce(row["TPERIPH_FL_DATA_SUSTRAVEL"], row["FL_SUSVELOCITY"], flparams),
          return_dtype=pl.Float64
      )
      .alias("FL_SUSFORCE"),

    pl.struct(["TPERIPH_FR_DATA_SUSTRAVEL", "FR_SUSVELOCITY"])
      .map_elements(
          lambda row: suspensionForce(row["TPERIPH_FR_DATA_SUSTRAVEL"], row["FR_SUSVELOCITY"], frparams),
          return_dtype=pl.Float64
      )
      .alias("FR_SUSFORCE"),
])

plt.plot(rd['Time_ms'], rd['TPERIPH_BL_DATA_SUSTRAVEL'], label='BL Suspension Travel')
plt.plot(rd['Time_ms'], rd['BL_SUSFORCE'], label='BL Suspension Force')
plt.plot(rd['Time_ms'], rd['BR_SUSFORCE'], label='BR Suspension Force')
plt.plot(rd['Time_ms'], rd['FL_SUSFORCE'], label='FL Suspension Force')
plt.plot(rd['Time_ms'], rd['FR_SUSFORCE'], label='FR Suspension Force')
plt.plot(rd['Time_ms'], rd['BL_SUSVELOCITY'], label='BL Suspension Velocity')

plt.title('BL Suspension Travel and Velocity vs Time')
plt.xlabel('Time (ms)')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
