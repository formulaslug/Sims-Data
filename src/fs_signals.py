"""

definition of column names and constants for fs-3 blue max data

"""

# Time 

t = "time" # this is what timeCol/simpleTimeCol produce

# High-voltage pack / motor controller 

V = "ACC_POWER_PACK_VOLTAGE"
I = "SME_TEMP_BusCurrent"
busV = "SME_TEMP_DC_Bus_V"
busC = "SME_TEMP_BusCurrent"

# GPS

lat = "VDM_GPS_Latitude"
lon = "VDM_GPS_Longitude"
speed = "VDM_GPS_SPEED" # raw GPS speed

# IMU 
xA_uncorrected = "VDM_X_AXIS_ACCELERATION"
yA_uncorrected = "VDM_Y_AXIS_ACCELERATION"
zA_uncorrected = "VDM_Z_AXIS_ACCELERATION"
zG = "VDM_Z_AXIS_YAW_RATE"

# Drivetrain
rpm = "SME_TRQSPD_Speed"

# Wheel speeds (hall effect)
heFL = "TPERIPH_FL_DATA_WHEELSPEED"
heFR = "TPERIPH_FR_DATA_WHEELSPEED"
heBL = "TPERIPH_BL_DATA_WHEELSPEED"
heBR = "TPERIPH_BR_DATA_WHEELSPEED"

# Lap column name (we’ll create it)
LAP_COL = "Lap"

# coords of blue max 

blueMaxGPS_Square = (
    (-121.7330999, 38.5759097),
    (-121.7328352, 38.5757670),
) 
