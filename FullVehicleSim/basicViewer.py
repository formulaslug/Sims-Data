import polars as pl
import matplotlib.pyplot as plt

df = pl.read_parquet("FullVehicleSim/simulation_output.parquet")

cols = ["x", "y", "z", "vX", "vY", "vZ", "speed", 
                    "headingX", "headingY", "headingZ", 
                    "yawRate", "frontBrakeTemperature", "rearBrakeTemperature", 
                    "charge", "drag", "resistiveForces", 
                    "motorTorque", "motorForce", "netForce", 
                    "maxTraction", "wheelRotationsHZ", "motorRPM",
                    "motorRotationsHZ", "current", 
                    "maxWheelTorque", "maxPower", "power", 
                    "voltage", 
                    "frontBrakeForce", "rearBrakeForce", 
                    "frontBrakeHeating", "rearBrakeHeating", 
                    "frontBrakeCooling", "rearBrakeCooling",
                    "frontSlipAngle", "rearSlipAngle"]

plt.plot(df["time"], df["motorForce"], label="motorForce")
plt.plot(df["time"], df["frontBrakeForce"] + df["rearBrakeForce"], label="Brake Force")
plt.legend()
plt.show()
