import polars as pl
import numpy as np
import matplotlib.pyplot as plt

df = pl.read_parquet("../fs-data/FS-3/03162026/2_steeper_regen_curve.parquet").fill_null(strategy="forward").fill_null(strategy="backward")

psi = np.clip((df["ETC_STATUS_BRAKE_SENSE_VOLTAGE"] - 330) / (3300 - 660) * 2000, 0, 2000)
brake_pedal = df["ETC_STATUS_BRAKE_PEDAL_TRAVEL"] / 100
accel_pedal = df["ETC_STATUS_PEDAL_TRAVEL"]/100



# plt.plot(psi)
# plt.plot(brake_pedal)
# plt.show()

[x for x in df.columns if "TMAIN" in x]


coeff_of_friction = 0.3
brake_disk_radius = 0.09525
wheel_radius = 0.216

gear_ratio = 12/39

brake_ratio = 0.55 # rear

def brakePSI_toNewtons(psi:float) -> float:
    return psi * 1.23 * 2 * 4.448222 * 2# lb force to Newtons (normal force on the brake disk)
    # psi, brake caliper area, 2 calipers, lb to newtons, 2 wheels, coeff of friction, brake disc radius, wheel radius 

def front_brake_force(brake_pressure):
    return brakePSI_toNewtons(brake_pressure) * coeff_of_friction * brake_disk_radius / wheel_radius

def regen_torque(front_psi):
    fbf = front_brake_force(front_psi)
    rear_force = fbf * brake_ratio / (1 - brake_ratio)
    rear_wheel_torque = rear_force * wheel_radius
    motor_torque = rear_wheel_torque * gear_ratio
    return motor_torque

fast = np.vectorize(regen_torque)

psi_range = np.linspace(0, 2000, 500)

torque_range = fast(psi_range)

fs3_regen = (brake_pedal - accel_pedal) * 180
fs4_regen = (regen_torque(psi))

plt.plot(fs3_regen, label="FS3 Regen")
plt.plot(fs4_regen, label="FS4 Regen")
plt.legend()
plt.show()

plt.plot(fs4_regen, label="FS4 Regen")
plt.plot(fs4_regen-accel_pedal*180, label="FS4 Regen + accel reduction")
plt.legend()
plt.show()



plt.plot(df["Time_ms"]/1000, fs3_regen, label="FS3 Regen")
plt.plot(df["Time_ms"]/1000, fs4_regen, label="FS4 Regen")
plt.plot(df["Time_ms"]/1000, fs4_regen-accel_pedal*180, label="FS4 Regen + accel reduction")
plt.legend()
plt.show()
