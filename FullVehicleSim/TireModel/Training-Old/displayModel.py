import polars as pl

import matplotlib.pyplot as plt

import dumpling

train = pl.read_csv("slipAngleData.dat", separator="	", skip_rows=1, skip_rows_after_header=1)  #pl.read_csv("/Users/daniel/Documents/Coding/FormulaSlug/sim-test/VehicleDynamics/TireModel/Training/train-slipangle.csv", separator=",")
#train = pl.read_csv("train.csv")
#train = pl.read_csv("train-combinedSlipLong.csv")
#train = train.filter((pl.col("FX") < 0) & (pl.col("ET") > 125) & (pl.col("ET") < 840))
#train = train.filter((pl.col("NFX") > -0.2) & (pl.col("NFX") < 0.2))
#train = train.filter((pl.col("SA") > -0.1) & (pl.col("SA") < 0.1))
#train = train.filter((pl.col("SR") > -1) & (pl.col("SR") < 0))

#mean_nfx = train.group_by("SA").agg(pl.col("FX").mean().alias("mean_FX"))

#train_with_mean = train.join(mean_nfx, on="SA")

#threshold = 0.5 # arbitrary filter value
#train = train_with_mean.filter(pl.col("FX") >= (pl.col("mean_FX") * threshold))

#da = da.with_columns((pl.col("FX") * 0.505).alias("T"))

def predictFX(FZ, SR, SA, V):

    scalars = {
    "shape-factor": 1.899999976158142,
    "stiffness-factor": 10.0,
    "curvature-scaling-factor": 1.2944897413253784,
    "horizontal-shift-factor": 1.0,
    "vertical-shift-factor": 1.0,
    "zeta_1": 1.6132731437683105,
    "p_px1": -0.34850001335144043,
    "p_px2": 0.37834998965263367,
    "p_dx1": 1.6569633483886719,
    "p_dx2": -0.08285000175237656,
    "p_dx3": 0.0,
    "p_dx4": 0.0,
    "p_ex1": 0.1111299991607666,
    "p_ex2": 0.314300000667572,
    "p_ex3": 0.0,
    "p_ex4": 0.0017190000507980585,
    "p_kx1": 21.687000274658203,
    "p_kx2": 13.727999687194824,
    "p_kx3": -0.4097999930381775,
    "p_hx1": 0.00021615000150632113,
    "p_hx2": 0.0011597999837249517,
    "p_vx1": 2.028299968515057e-05,
    "p_vx2": 0.00010568000288913026,
    "p_vx1": 2.028299968515057e-05,
    "p_vx2": 0.00010568000288913026,
    "r_bx1": 13.187339782714844,
    "r_bx2": 9.575895309448242,
    "r_bx3": 0.0,
    "r_cx1": 1.1033161878585815,
    "r_ex1": -0.5640894174575806,
    "r_ex2": -0.46630001068115234,
    "r_hx1": 0.10596346855163574,
    "lambda_xalpha": 1.1376594305038452,
    "lambda_alphastar": 1.1374865770339966,
    "lambda_combinedslipcoeff": -0.00019185709243174642,
    "combined_long_offset": -0.040454503148794174,
    "By1": 0.17617739737033844,
    "Cy1": 1.0135650634765625,
    "Dy1": 3.036080837249756,
    "Ey1": 0.8220884799957275,
    "e_one": -3.0873866081237793,
    "e_two": 3.0867669582366943,
    "e_three": 3.0867669582366943
    }
    
    Constants =  {
        "g": 9.8067,
        "e": 2.7183
    }

    MechanicalParameters = {
        "friction-coeff-lat": 1.7333,
        "friction-coeff-long": 1.7333,
        "unloaded-radius": 1.7333,
        "p_0": 82000,
        "load_0": 300
    }

    #print(FZ, SA, V)

    curr = dumpling.Tire(FZ, SR, SA, V, Constants, MechanicalParameters, scalars)
    #return SR * 100 #dumplingTest.Tire(SR)
    
    force =  curr.getLongForceCombinedSlip() * -1

    return force


train = train.with_columns(pl.lit(0.0).alias("predictFX"))

count = 0

#latForceArray = []

#for row in train.iter_rows(named=True):
#    count += 1
#    currentLatForce = predictFX(row["FZ"], row["SR"], row["SA"], row["V"]) / row["FZ"]
#    latForceArray.append( currentLatForce)
    #row["predictFX"] = currLongForce

#print(count)
#print(train)

fig = plt.figure()
#plt.scatter(train["SA"], train["NFX"], s = 1, color="red")
#plt.scatter(train["SA"], latForceArray , s = 1, color="orange")
plt.scatter(train["ET"], train["V"], s = 1, color="orange")
#plt.scatter(dc["SR"], dc["FX"], s = 1, color="yellow")
#plt.scatter(dd["SR"], dd["FX"], s = 1, color="green")
#plt.scatter(de["SR"], de["FX"], s = 1, color="blue")
#plt.scatter(df["SR"], df["FX"], s = 1, color="purple")
plt.show()

