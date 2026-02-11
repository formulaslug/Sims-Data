## Efficiency = Power at Motor/Power from Acc/MC
## Power at motor = 

import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from Data.FSLib.IntegralsAndDerivatives import *
from Data.FSLib.fftTools import *

cols = ['Seconds', ':Lap', ':LapTime', ':Time', 'APPS_1', 'APPS_2', 'APPS_Travel_1', 'APPS_Travel_2', 'Avg_Cell_Temp', 'BMS_Balancing', 'BMS_Fault', 'Brakes', 'CAN_Flag', 'Charging', 'Cockpit_Switch', 'Fans:Value', 'Fans_On', 'GLV_Voltage', 'Max_Cell_Temp', 'Motor_On', 'Pedal_Travel', 'Precharge:Value', 'Precharge_Done', 'RTDS_Queue', 'SME_CURRLIM_ChargeCurrentLim', 'SME_CURRLIM_DischargeCurrentLim', 'SME_TEMP_BusCurrent', 'SME_TEMP_ControllerTemperature', 'SME_TEMP_DC_Bus_V', 'SME_TEMP_FaultCode', 'SME_TEMP_FaultLevel', 'SME_TEMP_MotorTemperature', 'SME_THROTL_Forward', 'SME_THROTL_MBB_Alive', 'SME_THROTL_MaxSpeed', 'SME_THROTL_PowerReady', 'SME_THROTL_Reverse', 'SME_THROTL_TorqueDemand', 'SME_TRQSPD_Controller_Overtermp', 'SME_TRQSPD_Forward', 'SME_TRQSPD_Hydraulic', 'SME_TRQSPD_Key_switch_overvolt', 'SME_TRQSPD_Key_switch_undervolt', 'SME_TRQSPD_MotorFlags', 'SME_TRQSPD_Park_Brake', 'SME_TRQSPD_Pedal_Brake', 'SME_TRQSPD_Powering_Enabled', 'SME_TRQSPD_Powering_Ready', 'SME_TRQSPD_Precharging', 'SME_TRQSPD_Reverse', 'SME_TRQSPD_Running', 'SME_TRQSPD_SOC_Low_Hydraulic', 'SME_TRQSPD_SOC_Low_Traction', 'SME_TRQSPD_Speed', 'SME_TRQSPD_Torque', 'SME_TRQSPD_Traction', 'SME_TRQSPD_contactor_closed', 'Seg0_TEMP_0', 'Seg0_TEMP_1', 'Seg0_TEMP_2', 'Seg0_TEMP_3', 'Seg0_TEMP_4', 'Seg0_TEMP_5', 'Seg0_TEMP_6', 'Seg0_VOLT_0', 'Seg0_VOLT_1', 'Seg0_VOLT_2', 'Seg0_VOLT_3', 'Seg0_VOLT_4', 'Seg0_VOLT_5', 'Seg0_VOLT_6', 'Seg1_TEMP_0', 'Seg1_TEMP_1', 'Seg1_TEMP_2', 'Seg1_TEMP_3', 'Seg1_TEMP_4', 'Seg1_TEMP_5', 'Seg1_TEMP_6', 'Seg1_VOLT_0', 'Seg1_VOLT_1', 'Seg1_VOLT_2', 'Seg1_VOLT_3', 'Seg1_VOLT_4', 'Seg1_VOLT_5', 'Seg1_VOLT_6', 'Seg2_TEMP_0', 'Seg2_TEMP_1', 'Seg2_TEMP_2', 'Seg2_TEMP_3', 'Seg2_TEMP_4', 'Seg2_TEMP_5', 'Seg2_TEMP_6', 'Seg2_VOLT_0', 'Seg2_VOLT_1', 'Seg2_VOLT_2', 'Seg2_VOLT_3', 'Seg2_VOLT_4', 'Seg2_VOLT_5', 'Seg2_VOLT_6', 'Seg3_TEMP_0', 'Seg3_TEMP_1', 'Seg3_TEMP_2', 'Seg3_TEMP_3', 'Seg3_TEMP_4', 'Seg3_TEMP_5', 'Seg3_TEMP_6', 'Seg3_VOLT_0', 'Seg3_VOLT_1', 'Seg3_VOLT_2', 'Seg3_VOLT_3', 'Seg3_VOLT_4', 'Seg3_VOLT_5', 'Seg3_VOLT_6', 'Shutdown:Value', 'Shutdown_Closed', 'TELEM_BL_SUSTRAVEL', 'TELEM_BR_SUSTRAVEL', 'TELEM_FL_SUSTRAVEL', 'TELEM_FR_SUSTRAVEL', 'TELEM_STEERBRAKE_BRAKEF', 'TELEM_STEERBRAKE_BRAKER', 'TELEM_STEERBRAKE_STEER', 'TS_Current', 'TS_Ready', 'TS_Voltage', 'VDM_GPS_ALTITUDE', 'VDM_GPS_Latitude', 'VDM_GPS_Longitude', 'VDM_GPS_SATELLITES_IN_USE', 'VDM_GPS_SPEED', 'VDM_GPS_TRUE_COURSE', 'VDM_GPS_VALID1', 'VDM_GPS_VALID2', 'VDM_UTC_DATE_DAY', 'VDM_UTC_DATE_MONTH', 'VDM_UTC_DATE_YEAR', 'VDM_UTC_TIME_HOURS', 'VDM_UTC_TIME_MINUTES', 'VDM_UTC_TIME_SECONDS', 'VDM_X_AXIS_ACCELERATION', 'VDM_X_AXIS_YAW_RATE', 'VDM_Y_AXIS_ACCELERATION', 'VDM_Y_AXIS_YAW_RATE', 'VDM_Z_AXIS_ACCELERATION', 'VDM_Z_AXIS_YAW_RATE']

df = pl.read_parquet("FS-2/Parquet/2025-03-06-BrakingTests1.parquet").select(cols).vstack(
    pl.read_parquet("FS-2/Parquet/2025-03-06-Part2.parquet").select(cols)).vstack(
    pl.read_parquet("FS-2/Parquet/2024-12-02-Part1-100Hz.pq")).vstack(
    pl.read_parquet("FS-2/Parquet/2024-12-02-Part2-100Hz.pq")).vstack(
    pl.read_parquet("FS-2/Parquet/2025-01-17-P1.parquet").select(cols)).vstack(
    pl.read_parquet("FS-2/Parquet/2025-01-17-P2.parquet").select(cols)).vstack(
    pl.read_parquet("FS-2/Parquet/2025-02.parquet").select(cols)).vstack(
    pl.read_parquet("FS-2/Parquet/2025-03-05.parquet").select(cols))
df.columns

brakeF = 'TELEM_STEERBRAKE_BRAKEF'
brakeR = 'TELEM_STEERBRAKE_BRAKER'
torque = 'SME_TRQSPD_Torque'
torqueDemand = 'SME_THROTL_TorqueDemand'

speed = "VDM_GPS_SPEED"
mps_speed = "VMD_GPS_Speed_mps"
rpm = "SME_TRQSPD_Speed"

curAcc = "TS_Current"
vAcc = "TS_Voltage"

curBus = "SME_TEMP_BusCurrent"
vBus = "SME_TEMP_DC_Bus_V"

xA_uncorrected = "VDM_X_AXIS_ACCELERATION"
yA_uncorrected = "VDM_Y_AXIS_ACCELERATION"
zA_uncorrected = "VDM_Z_AXIS_ACCELERATION"
vA_uncorrected = "vA_uncorrected"

xA = "xA"
yA = "yA"
zA = "zA"
motor_mph = "motor mph"

power_mech = "power_mech"
power_electrical = "power_electrical"

df.insert_column(df.get_column_index(speed), ((df.select(speed))[speed]*0.44704).alias("VMD_GPS_Speed_mps"))
df.insert_column(-1, (df[xA_uncorrected].pow(2) + df[yA_uncorrected].pow(2) + df[zA_uncorrected].pow(2)).sqrt().alias("vA_uncorrected")) # Column that is magnitude of acceleration
df.insert_column(-1, (df[torque] * df[rpm]/60*2*np.pi).alias("power_mech"))
df.insert_column(-1, (df[curAcc] * df[vAcc]).alias("power_electrical"))
df.insert_column(-1, (df[rpm]*0.013008).alias("motor mph"))


mass = 0.4535924*(509 + 145) # Car Weight Lbs, Daniel Weight Lbs
car_imu_args = np.asarray([ 0.96324388,  0.98596328,  0.96496715,  0.14328537, -0.25890404,
       -0.11429213,  0.12046681,  0.27326771, -0.08613647,  0.03362676,
       -0.02482835, -0.02720422])
Sx, Sy, Sz, a1, a2, a3, a4, a5, a6, b1, b2, b3 = tuple(car_imu_args)

b1 = df.filter(pl.col(vA_uncorrected) < 1.03).filter(pl.col(vA_uncorrected) > 0.97)[xA_uncorrected].mean()
b2 = df.filter(pl.col(vA_uncorrected) < 1.03).filter(pl.col(vA_uncorrected) > 0.97)[yA_uncorrected].mean()
b3 = df.filter(pl.col(vA_uncorrected) < 1.03).filter(pl.col(vA_uncorrected) > 0.97)[zA_uncorrected].mean()

def apply_correction_acc_Gs (df, Sx, Sy, Sz, a1, a2, a3, a4, a5, a6, b1, b2, b3): #Apply the correction matrices on the dataframe and update it
    scalar_matrix = np.array([[Sx, 0, 0],
                              [0, Sy, 0],
                              [0, 0, Sz]])
    off_axis_matrix = np.array([[1, a1, a2],
                                [a3, 1, a4],
                                [a5, a6, 1]])
    first_matrix = np.matmul(scalar_matrix, off_axis_matrix)
    bias1 = np.ones((1,df[xA_uncorrected,yA_uncorrected,zA_uncorrected].shape[0]))*b1
    bias2 = np.ones((1,df[xA_uncorrected,yA_uncorrected,zA_uncorrected].shape[0]))*b2
    bias3 = np.ones((1,df[xA_uncorrected,yA_uncorrected,zA_uncorrected].shape[0]))*b3
    bias_matrix = np.concatenate([bias1,bias2,bias3], axis=0)
    matrix = df[xA_uncorrected,yA_uncorrected,zA_uncorrected].to_numpy().T
    biased_matrix = matrix-bias_matrix
    vectors = np.matmul(first_matrix,biased_matrix)
    print(vectors)
    df = df.insert_column(-1, pl.Series(vectors[0,:]).alias("xA"))
    df = df.insert_column(-1, pl.Series(vectors[1,:]).alias("yA"))
    df = df.insert_column(-1, pl.Series(vectors[2,:]).alias("zA"))
    return df

_ = apply_correction_acc_Gs(df, Sx, Sy, Sz, a1, a2, a3, a4, a5, a6, b1, b2, b3)

# plt.plot(df[mps_speed])
# plt.plot(in_place_derive(df[mps_speed]))
# plt.plot(in_place_integrate(df[xA]*9.807))
# plt.plot(df[xA]*9.807)
# plt.legend(["Speed", "Acceleration", "SpeedIMU", "AccelerationIMU"])

driving = df.filter(pl.col(rpm) > 25)

# driving = df.filter(
#     pl.col("SME_THROTL_TorqueDemand") > 29000).filter(
#     pl.col("SME_TRQSPD_Torque") > 5).filter(
#     pl.col(power_electrical) > 5)
# # driving = df.filter(pl.col(rpm) > 20).filter(pl.col(brakeF) < 7500).filter(pl.col("SME_THROTL_TorqueDemand") > 0).filter(pl.col(xA) > 0)
has_power = driving.filter(pl.col(power_mech) > 100)
print(driving.shape)

# power_mech = mass * driving[xA]*9.8 * driving[mps_speed]
# power_mech = driving[torque] * driving[rpm]/60*2*np.pi
# power_electrical = driving[curAcc] * driving[vAcc]

# plt.plot(low_pass_filter(df[brakeF].to_numpy(), 0.9))
# plt.plot(low_pass_filter(df[brakeR].to_numpy(), 0.9))

df = df.sample(400000)

# plt.scatter(df[curAcc]*df[vAcc], df[curBus]*df[vBus], s=0.3)
# plt.legend(["Accumulator Power", "Bus/Motor Power"])
# plt.xlabel("Accumulator Power (Watts)")
# plt.ylabel("Motor Controller Power (Watts)")
# plt.ylim(0, 40000)
# plt.xlim(0, 40000)
# plt.plot([0,20000,40000], [0,20000,40000], color='red')
# plt.scatter(driving[rpm], driving[power_mech], c=driving[torqueDemand], s=0.3,cmap='prism', alpha=0.1)
# plt.scatter(df[rpm], df[power_electrical],s=0.3)
# plt.scatter(df[rpm], df[power_electrical]/(df[rpm]/30*np.pi), s=0.3)


# plt.scatter(df[rpm], df[power_electrical],c=df[power_electrical],cmap="viridis",s=0.3)
# plt.scatter(df[rpm], df[torque], s=0.3)
# plt.scatter(df[rpm], df[torque], c=df[torqueDemand], s=0.3,cmap='prism', alpha=0.1)
# plt.scatter(driving[rpm], power_electrical,c=driving[torqueDemand],  s=0.3)
# plt.legend(["Accumulator Torque", "Torque"])

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax2 = ax.twinx()
ax.scatter(driving[rpm], driving[power_electrical]/(driving[rpm]/30*np.pi), s=0.3)
ax2.scatter(driving[rpm], driving[power_electrical],s=0.3,color='red',alpha=0.01)
ax.set_xlabel("rpm")
ax.set_ylabel("Torque")
ax2.set_ylabel("Power")
fig.legend(["Accumulator Torque", "Accumulator Power"])



# plt.ylim(0,500)
# plt.scatter(power_mech, power_electrical, s=0.3)
# plt.plot(((0,30000)), ((0,30000)))

print(df[power_electrical].mean())
plt.show()
