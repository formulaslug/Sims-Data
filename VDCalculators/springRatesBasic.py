import numpy as np
import matplotlib.pyplot as plt



mass = 293.97 #FS-3 Comp weight + 160lb driver, DOES NOT INCLUDE AERO PACKAGE
weight = 2883.8457  # N
frontWD = 0.4632
rearWD = 1 - frontWD
leftWD = 0.492
CGHeight = 0.234  # meters
trackWidth = 1.234
wheelBase = 1.59
RCFront = 0.0203
RCRear = 0.0493
#recalcualate RC's after onshape is done
#Right now rear is ~0.0426974 mm
motionRatioF = 1.006
motionRatioR = 1.004
multiplier = 0.00571015 #n/m -> lbf/in
masterAy= 1.7
masterV = 30
TRG = 0.01524409115   # target roll gradient in rad/g
print("TRG = ", TRG)
###     Aero Loads (N)  ###


def compute_spring_rates(v, ay, trg):
    frontWing = 0.88888 * v**2
    rearWing = 1.111 * v**2     #Taken from total downforce goal
    floorTotal = 0.22222 * v**2    #with the distribution from the fs-3 
                    #aero package CFD
# Split floor 50 - 50
    frontFloor = floorTotal * 0.5
    rearFloor = floorTotal * 0.5

    # frontAero = frontFloor + frontWing
    # rearAero = rearFloor + rearWing

    frontAero = 0
    rearAero = 0

    print("Aero Balance (front%) =", frontAero/2000)
    print("Compute_spring_rates is running")
  


    ###     Axle Weights    ###

    frontAW = frontWD * weight + frontAero
    rearAW = rearWD * weight + rearAero

    ###     Roll Moments    ###

    frontRM = frontAW * (CGHeight - RCFront) * ay
    rearRM = rearAW * (CGHeight - RCRear) * ay

    ###     Finding Required Roll stiffness     ###

    frontRS = frontRM / trg
    rearRS = rearRM / trg

    ###     Moment from one side about CG       ###
    # roll = Φ
    # track width = t
    # Δz = Φ * t/2
    # wheel rate per side = kw
    # Force at each wheel:
    #     F = kw * Φ * t/2
    # Moment from one side:
    #     Mside = F * (t/2) = kw * Φ * (t/2)^2

    # (t/2)^2 term
    halfTrackSq = (trackWidth / 2)**2

    ###     Wheel Rates    ###
    # Kphi = 2 * kw * (t/2)^2
    # kw = Kphi / (2 * (t/2)^2)

    frontKW = frontRS / (2 * halfTrackSq)
    rearKW = rearRS / (2 * halfTrackSq)

    ###     Spring Rates (using motion ratios)     ###
    # kw = ks * MR^2
    # ks = kw / MR^2

    frontKS = frontKW / (motionRatioF**2)
    rearKS = rearKW / (motionRatioR**2)
    return frontKS, rearKS, frontRS, rearRS, frontKW, rearKW

def roll_and_pitch_gradients(v, ay, ax, frontKS, rearKS):
    """
    Returns:
    roll_gradient  (deg/g)
    pitch_gradient (deg/g)
    """

    # --- Convert springs to wheel rates ---
    frontKW = frontKS * motionRatioF**2
    rearKW  = rearKS  * motionRatioR**2

    # ================= ROLL =================
    halfTrackSq = (trackWidth / 2)**2
    frontRS = 2 * frontKW * halfTrackSq
    rearRS  = 2 * rearKW  * halfTrackSq
    Kphi = frontRS + rearRS

    # Aero
    frontAero = 0.88888 * v**2 + 0.5 * 0.22222 * v**2
    rearAero  = 1.111   * v**2 + 0.5 * 0.22222 * v**2

    frontAW = frontWD * weight + frontAero
    rearAW  = rearWD  * weight + rearAero

    M_roll = (
        frontAW * (CGHeight - RCFront) +
        rearAW  * (CGHeight - RCRear)
    ) * ay

    phi = M_roll / Kphi
    roll_gradient = np.degrees(phi / ay)

    # ================= PITCH =================
    af = wheelBase * rearWD
    ar = wheelBase * frontWD

    PCHeight = 0.5 * (RCFront + RCRear)

    M_pitch = weight * (CGHeight - PCHeight) * ax

    Ktheta = frontKW * af**2 + rearKW * ar**2

    theta = M_pitch / Ktheta
    pitch_gradient = np.degrees(theta / ax)

    return roll_gradient, pitch_gradient


###     Graphs      ###
def spring_rates_vs_speed_fixed_g():
    speeds = np.linspace(10, 35, 30)
    ay = 2.2
    front_rates = []
    rear_rates = []
    for v in speeds:
        frontKS, rearKS, _, _ = compute_spring_rates(v, ay, TRG)
        front_rates.append(frontKS / 1000)  # N/mm
        rear_rates.append(rearKS / 1000)

    plt.plot(speeds, front_rates, label="Front")
    plt.plot(speeds, rear_rates, label="Rear")

    plt.xlabel("Speed (m/s)")
    plt.ylabel("Spring Rate (N/mm)")
    plt.title("Required Spring Rate vs Speed @ 1.7g")
    plt.legend()
    plt.grid(True)
    plt.show()


def roll_vs_lateral():
    g_vals = np.linspace(0.5, 2.5, 30)
    v = 30  # m/s

    roll_angles = []

    for ay in g_vals:
        _, _, frontRS, rearRS = compute_spring_rates(v, ay, TRG)
        Kphi_total = frontRS + rearRS
        phi = ay / TRG   # radians
        roll_angles.append(np.degrees(phi))

    plt.plot(g_vals, roll_angles)
    plt.xlabel("Lateral Acceleration (g)")
    plt.ylabel("Roll Angle (deg)")
    plt.title("Roll Angle vs Lateral G @ 30 m/s")
    plt.grid(True)
    plt.show()

def roll_angle_vs_speed_calc(speeds, ay, frontKS, rearKS):
    """
    Compute roll angle (deg) vs speed for fixed lateral g and fixed springs.

    speeds  : array-like, vehicle speed (m/s)
    ay      : lateral acceleration (g)
    frontKS : front spring rate (N/m)
    rearKS  : rear spring rate (N/m)

    Returns:
    roll_angles : numpy array of roll angle in degrees
    """

    # --- Convert springs to roll stiffness ---
    frontKW = frontKS * motionRatioF**2
    rearKW  = rearKS  * motionRatioR**2

    halfTrackSq = (trackWidth / 2)**2

    frontRS = 2 * frontKW * halfTrackSq
    rearRS  = 2 * rearKW  * halfTrackSq

    Kphi_total = frontRS + rearRS

    roll_angles = []

    for v in speeds:

        # --- Aero forces ---
        frontWing = 0.88888 * v**2
        rearWing  = 1.111   * v**2
        floorTotal = 0.22222 * v**2

        frontAero = frontWing + 0.5 * floorTotal
        rearAero  = rearWing  + 0.5 * floorTotal

        # --- Axle loads ---
        frontAW = frontWD * weight + frontAero
        rearAW  = rearWD  * weight + rearAero

        # --- Roll moment ---
        frontRM = frontAW * (CGHeight - RCFront) * ay
        rearRM  = rearAW  * (CGHeight - RCRear)  * ay

        M_roll = frontRM + rearRM

        # --- Roll angle ---
        phi = M_roll / Kphi_total      # radians
        roll_angles.append(np.degrees(phi))

    return np.array(roll_angles)

def roll_angle_vs_speed_display(ay):
    speeds = np.linspace(10, 35, 40)
    

    frontKS = compute_spring_rates(masterV, masterAy, TRG)[0]   # N/m
    rearKS  = compute_spring_rates(masterV, masterAy, TRG)[1]   # N/m
    print(frontKS, rearKS)
    roll = roll_angle_vs_speed_calc(speeds, ay, frontKS, rearKS)
    print(frontKS, rearKS)

    plt.plot(speeds, roll)
    plt.xlabel("Speed (m/s)")
    plt.ylabel("Roll Angle (deg)")
    plt.title(f"Roll Angle vs Speed @ {ay}g")
    plt.grid(True)
    plt.show()
#for roll angle vs speed at a set g, call the display one with a constant ay as g's
computed_spring_rates = compute_spring_rates(masterV,masterAy, TRG)
frontKS = computed_spring_rates[0]
rearKS = computed_spring_rates[1]
frontRS = computed_spring_rates[2]
rearRS = computed_spring_rates[3]
frontKW = computed_spring_rates[4]
rearKW = computed_spring_rates [5]

###     Print results     ###

print("Front Wheel Rate (kN/m):", frontKW/1000)
print("Rear Wheel Rate (kN/m):", rearKW/1000)
print("Front Spring Rate (N/m):", frontKS)
print("Rear Spring Rate (N/m):", rearKS)
print("Front Spring Rate (lbf/in):", frontKS * multiplier)
print("Rear Spring Rate (lbf/in):", rearKS * multiplier)
print()
roll_gradient = roll_and_pitch_gradients(30, 1, 1, frontKS, rearKS)[0]
pitch_gradient = roll_and_pitch_gradients(30, 1, 1, frontKS, rearKS)[1]
print("roll gradient:", roll_gradient*0.0174533, "rads/g")
print(roll_gradient)
print("pitch gradient:", pitch_gradient*0.0174533, "rads/g")
print(pitch_gradient)

