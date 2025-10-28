from xgboost import XGBClassifier
import polars as pl
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from Data.AnalysisFunctions import *
import matplotlib.pyplot as plt

t = "Time"
mcVoltage = "SME_TEMP_DC_Bus_V"
accVoltage = "ACC_POWER_PACK_VOLTAGE"
accVoltage = "TS_Voltage"
faultCode = "SME_TEMP_FaultCode"
prechargeDone = 'ACC_STATUS_PRECHARGE_DONE'
precharging = 'ACC_STATUS_PRECHARGING'
pwrReady = "SME_THROTL_PowerReady"
torqueDemand = "SME_THROTL_TorqueDemand"
mcOvertemp = "SME_TRQSPD_Controller_Overtermp"
df = read("../fs-data/FS-3/08172025/08172025_22_6LapsAndWeirdCurrData.parquet")
df = read("../fs-data/FS-3/10082025/fixed_wheels_nathaniel_inv_test_w_fault.parquet")
df = read("../fs-data/FS-3/08102025/08102025Endurance1_SecondHalf.parquet")
df = read("../fs-data/FS-3/08102025/08102025RollingResistanceTestP3.parquet")
df = read("../fs-data/FS-3/08172025/08172025_26autox1.parquet")
# df = read("C:/Projects/FormulaSlug/fs-data/FS-2/Parquet/2025-01-14.parquet")

df.columns

plt.plot(df[mcVoltage], label = mcVoltage)
plt.plot(df[accVoltage], label = accVoltage)
plt.plot(df[faultCode], label = faultCode)
plt.plot(df[prechargeDone] * 50, label = prechargeDone)
plt.plot(df[precharging] * 40, label = precharging)
plt.plot(df[pwrReady] * 30, label = pwrReady)
plt.plot(df[torqueDemand] / 100, label = torqueDemand)
plt.plot(df[mcOvertemp] * 20, label = mcOvertemp)
plt.suptitle("08102025Endurance1_SecondHalf")
plt.legend()
plt.show()

[x for x in df.columns if "SME_" in x]

# fs-data/FS-3/08102025/08102025Endurance1_SecondHalf.parquet @ 970951 -- Not this one. Just Driving.
# fs-data/FS-3/08102025/08102025RollingResistanceTestP2.parquet @ 98591 -- Car being turned off
# fs-data/FS-3/08172025/08172025_22_6LapsAndWeirdCurrData.parquet @ 518538, 560592, 685456
# fs-data/FS-3/08102025/08102025RollingResistanceTestP3.parquet @ 111263
# fs-data/FS-3/08102025/08102025RollingResistanceTestP1.parquet @ 201624
# fs-data/FS-3/08102025/08102025RollingResistanceTestP4.parquet @ 33955
# fs-data/FS-3/08172025/08172025_26autox1.parquet @ 123328, 313745
# fs-data/FS-3/10082025/fixed_wheels_nathaniel_inv_test_w_fault.parquet @ 320629

data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(data['data'], data['target'], test_size=.2) #type: ignore
bst = XGBClassifier(n_estimators=2, max_depth=2, learning_rate=1, objective='binary:logistic')
bst.fit(X_train, y_train)
preds = bst.predict(X_test)