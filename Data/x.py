import scipy
import scipy.integrate

batemoData = scipy.io.loadmat("C:\\Users\\Goob\\Downloads\\OneDrive_2026-01-07\\Batemo Sponsorship\\Batemo Cell Data Molicel INR18650P30B\\Batemo Cell Data Package\\Molicel_INR18650P30B_measurement.mat",
                              simplify_cells=True)


header = batemoData["__header__"]
version = batemoData["__version__"]
globals = batemoData["__globals__"] ## Nothing
print(f"header = {header}")
print(f"version = {version}")
measurement = batemoData["measurement"]
firstLayerMeta = measurement['meta']
Fu = measurement['fu']

## DCC (Discharge)
## CHC (Charging)
## DCP (Discharge Pulse)
## CHP (Charge Pulse)
## PRO (Profile Measurement)


# Each Has
    # name
    # T_amb (Ambient Temperature)
    # t (Time Seconds)
    # I (Current)
    # V (Voltage)
    # T_surf (Surface temperature, nominally 1xN but may be 2xN?)


for i in Fu['DCC']:
    print(i['name'])