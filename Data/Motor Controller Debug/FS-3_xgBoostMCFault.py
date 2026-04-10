from xgboost import XGBClassifier
from xgboost import plot_tree
import polars as pl
# from sklearn.datasets import load_iris
# from sklearn.model_selection import train_test_split
from Data.FSLib.AnalysisFunctions import *
import matplotlib.pyplot as plt
from explainerdashboard import ClassifierExplainer, ExplainerDashboard

t = "Time"
mcVoltage = "SME_TEMP_DC_Bus_V"
mcCurrent = "SME_TEMP_BusCurrent"
accVoltage = "ACC_POWER_PACK_VOLTAGE"
accVoltage = "TS_Voltage"
faultCode = "SME_TEMP_FaultCode"
prechargeDone = 'ACC_STATUS_PRECHARGE_DONE'
precharging = 'ACC_STATUS_PRECHARGING'
pwrReady = "SME_THROTL_PowerReady"
torqueDemand = "SME_THROTL_TorqueDemand"
mcOvertemp = "SME_TRQSPD_Controller_Overtermp"
mcTemp = "SME_TEMP_ControllerTemperature"
mTemp = "SME_TEMP_MotorTemperature"
bmsFault = "BMS_Fault"
mOn = "Motor_On"

df = read("../fs-data/FS-3/08172025/08172025_22_6LapsAndWeirdCurrData.parquet")
df = read("../fs-data/FS-3/10082025/fixed_wheels_nathaniel_inv_test_w_fault.parquet")
df = read("../fs-data/FS-3/08102025/08102025Endurance1_SecondHalf.parquet")
df = read("../fs-data/FS-3/08102025/08102025RollingResistanceTestP3.parquet")
df = read("../fs-data/FS-3/08172025/08172025_26autox1.parquet")
df = read("../fs-data/FS-2/Parquet/2025-03-06-Part2.parquet") # a few examples with temp > 44
df = read("C:/Projects/FormulaSlug/fs-data/FS-2/Parquet/2024-12-02-Part1-100Hz.pq") # 1 example with temp > 44
df = read("C:/Projects/FormulaSlug/fs-data/FS-2/Parquet/2024-12-02-Part2-100Hz.pq") # Nothing
df = read("C:/Projects/FormulaSlug/fs-data/FS-2/Parquet/2025-01-14.parquet")
# df = read("C:/Projects/FormulaSlug/fs-data/FS-2/Parquet/2025-01-14.parquet")

df.columns

plt.plot(df[mcVoltage], label = mcVoltage)
plt.plot(df[accVoltage], label = accVoltage)
plt.plot(df[faultCode], label = faultCode)
plt.plot(df[mcCurrent], label = mcCurrent)
# plt.plot(df[prechargeDone] * 50, label = prechargeDone)
# plt.plot(df[precharging] * 40, label = precharging)
# plt.plot(df[pwrReady] * 30, label = pwrReady)
# plt.plot(df[torqueDemand] / 100, label = torqueDemand)
# plt.plot(df[mcOvertemp] * 20, label = mcOvertemp)
plt.plot(df[mcTemp], label = mcTemp)
plt.plot(df[mTemp], label = mTemp)
plt.plot(df[bmsFault] * 30, label = bmsFault)
plt.plot(df[mOn] * 20, label = mOn)
# plt.plot(df[])
# plt.suptitle("08102025Endurance1_SecondHalf")
plt.legend()
plt.show()

df.filter(pl.col(mcTemp) >= 44).filter(pl.col("Motor_On") == 1)["Label"].mean()

df["SME_TRQSPD_MotorFlags"]

[x for x in df.columns if "SME_" in x]

# fs-data/FS-3/08102025/08102025Endurance1_SecondHalf.parquet @ 970951 -- Not this one. Just Driving.
# fs-data/FS-3/08102025/08102025RollingResistanceTestP2.parquet @ 98591 -- Car being turned off
# fs-data/FS-3/08172025/08172025_22_6LapsAndWeirdCurrData.parquet @ 518538, 560592, 685456
# fs-data/FS-3/08102025/08102025RollingResistanceTestP3.parquet @ 111263
# fs-data/FS-3/08102025/08102025RollingResistanceTestP1.parquet @ 201624
# fs-data/FS-3/08102025/08102025RollingResistanceTestP4.parquet @ 33955
# fs-data/FS-3/08172025/08172025_26autox1.parquet @ 123328, 313745
# fs-data/FS-3/10082025/fixed_wheels_nathaniel_inv_test_w_fault.parquet @ 320629

df = df.with_columns([
    (pl.col(faultCode) == 26).cast(pl.Int32).alias("Label")
])

df.columns = [x.replace(".", "_") for x in df.columns]

drops = ["Label", smeFaultCode, smeFaultLevel, busV, "Seconds", lat, long, course, speed, vdmValid, "VDM_GPS_UTC_TIME", xA, yA, zA, vA, xA_uncorrected, yA_uncorrected, zA_uncorrected, xA_mps, yA_mps, zA_mps, xG, yG, zG, frT, flT, brT, blT, heFL, heFR, heBL, heBR, pedalTravel, etcBrakeVoltage, brakeF, brakeR]
drops = [c for c in drops if c in df.columns]

# Drop MotorFlags too
if "SME_TRQSPD_MotorFlags" in df.columns:
    drops.append("SME_TRQSPD_MotorFlags")

data = df.drop(*drops)
# X_train, X_test, y_train, y_test = train_test_split(data.drop(*drops), data['Label'], test_size=.2) #type: ignore
bst = XGBClassifier(n_estimators=5, max_depth=5, learning_rate=1, objective='binary:logistic')
bst.fit(data, df['Label'])
preds = bst.predict(data)
# bst.get_booster().feature_names = ["f"+str(i) for i in range(data.width)]
bst.get_booster().feature_names = data.columns

# Get feature names after drops (polars DataFrame)
_feature_cols = data.columns

# Weight importances (counts)
_weight_scores = bst.get_booster().get_score(importance_type='weight') or {}
if _weight_scores:
    print("Feature importances (weight) sorted:")
    for rank, (k, v) in enumerate(sorted(zip(["f"+str(i) for i in range(data.width)], _weight_scores.values()), key=lambda kv: kv[1], reverse=True), start=1):
        feat_idx = int(k[1:])
        feat_name = _feature_cols[feat_idx] if feat_idx < len(_feature_cols) else "<unknown>"
        print(f"{rank}. {k}: {v} -- Feature {feat_idx} : {feat_name}")
else:
    print("No weight importances found.")

# Gain importances (average gain)
_gain_scores = bst.get_booster().get_score(importance_type='gain') or {}
if _gain_scores:
    print("\nFeature importances (gain) sorted:")
    for rank, (k, v) in enumerate(sorted(zip(["f"+str(i) for i in range(data.width)], _gain_scores.values()), key=lambda kv: kv[1], reverse=True), start=1):
        feat_idx = int(k[1:])
        feat_name = _feature_cols[feat_idx] if feat_idx < len(_feature_cols) else "<unknown>"
        print(f"{rank}. {k}: {v:.6f} -- Feature {feat_idx} : {feat_name}")
else:
    print("No gain importances found.")

explainer = ClassifierExplainer(bst, data.to_pandas(), df['Label'].to_pandas())
db = ExplainerDashboard(explainer)
db.run()

explainer.dump('explainer.joblib')
# ClassifierExplainer.from_file('explainer.joblib')

fmapFile = pl.DataFrame([pl.Series(np.arange(df.width)), pl.Series(df.columns), pl.Series(["q" if df.filter(pl.col(a) != 1).filter(pl.col(a) != 0).height != 0 else "i" for a in df.columns])])
fmapFile.write_csv("xgboost_fmap.txt", separator="\t", include_header=True)

plot_tree(bst, num_trees=0, rankdir='LR')
plt.show()

df = pl.read_parquet("FS-3/10112025/firstDriveMCError30.parquet")
sorted(df.columns)

explainer.dump("explainer.joblib") 
## Identify events and label them
## Identify other times (equivalent number of negative events)
## Pull all input values from a half second before as features