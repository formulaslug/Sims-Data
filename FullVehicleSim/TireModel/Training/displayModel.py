import polars as pl

import matplotlib.pyplot as plt

import dumplingPure as dumpling

#train = pl.read_csv("slipAngleData.dat", separator="	", skip_rows=1, skip_rows_after_header=1)
train = pl.read_csv("Data/combinedDataSet.csv") 

def predictFX(FZ, SR, SA, V):

    scalars = {
    "shape-factor": 1.954084038734436,
    "stiffness-factor": 10.066422462463379,
    "curvature-scaling-factor": 1.213955283164978,
    "horizontal-shift-factor": 1.0,
    "vertical-shift-factor": 0.6640418767929077,
    "zeta_1": 0.7150991559028625,
    "p_px1": -0.34850001335144043,
    "p_px2": 0.37834998965263367,
    "p_dx1": 3.2977943420410156,
    "p_dx2": -0.0179620161652565,
    "p_dx3": 0.0,
    "p_dx4": 0.0,
    "p_ex1": 0.06139938533306122,
    "p_ex2": 0.34025561809539795,
    "p_ex3": 0.15006783604621887,
    "p_ex4": -0.07350758463144302,
    "p_kx1": 21.687000274658203,
    "p_kx2": 13.727999687194824,
    "p_kx3": -0.4097999930381775,
    "p_hx1": 0.00021615000150632113,
    "p_hx2": 0.0011597999837249517,
    "p_vx1": -0.0005141909932717681,
    "p_vx2": 0.03788552060723305,
    "lambda_loadscalar": 1.0094040632247925,
    "lambda_pressurescalar": 1.0,

    "r_bx1": 13.111207962036133,
    "r_bx2": 9.647605895996094,
    "r_bx3": 0.0,
    "r_cx1": 1.0115212202072144,
    "r_ex1": -0.4905780255794525,
    "r_ex2": -0.539811372756958,
    "r_hx1": 0.00888052023947239,
    "lambda_xalpha": 1.061378002166748,
    "lambda_alphastar": 1.1856805086135864,
    "lambda_combinedslipcoeff": 0.0013656612718477845,
    "combined_long_offset": 0.7030096650123596
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
    
    force =  curr.getLongForce() / FZ 
    return force


train = train.with_columns(pl.lit(0.0).alias("predictFX"))

count = 0

longForceArray = []

for row in train.iter_rows(named=True):
    count += 1
    currentLongForce = predictFX(row["FZ"], row["SR"], row["SA"], row["V"])
    longForceArray.append(currentLongForce)
    #row["predictFX"] = currLongForce

#print(count)
#print(train)

fig = plt.figure()
plt.scatter(train["SA"], train["NFY"], s = 1, color="red")
plt.scatter(train["SA"], longForceArray , s = 1, color="orange")
#plt.scatter(train["ET"], train["SR"], s = 1, color="orange")
#plt.scatter(dc["SR"], dc["FX"], s = 1, color="yellow")
#plt.scatter(dd["SR"], dd["FX"], s = 1, color="green")
#plt.scatter(de["SR"], de["FX"], s = 1, color="blue")
#plt.scatter(df["SR"], df["FX"], s = 1, color="purple")
plt.show()

