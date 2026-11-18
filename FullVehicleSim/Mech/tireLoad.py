import numpy as np  # noqa: N999

from ..paramLoader import *

# References: [Pacejka] H. B. Pacejka, Tire and Vehicle Dynamics, 3rd ed. (2012), Section 1.3.2, p. 20.
# Vehicle frame: x forward, y LEFT. Positive a_y / Fy = turning left = load moves to the RIGHT (outside) wheels.
# Simplifications: rigid chassis, whole mass treated as sprung, quasi-static roll (roll damping/inertia neglected).


def getRollStiffness(axle:str) -> float:
    # Suspension roll rate (chassis to wheel centre) from the DSS, Nm/deg -> Nm/rad. c_phi_i in [Pacejka] Eq. 1.34d.
    return Parameters[f"{axle}RollStiffness"] * 180.0 / np.pi

def getRollAxisHeightAtCG() -> float:
    # Roll axis = line through the front and rear roll centres; height under the CG (a from the front axle).
    front, rear = Parameters["frontRollCenterHeight"], Parameters["rearRollCenterHeight"]
    return front + (rear - front) * Parameters["a"] / Parameters["wheelBase"]

def calcRollAngle(accelerationY:float) -> float:
    """
    Quasi-static body roll, [Pacejka] Eq. 1.34d (p. 20) with all rates/accelerations of roll set to zero:
        (c_phi1 + c_phi2 - m*g*h') * phi = -m*h' * a_y      (h' = CG height above the roll axis)
    Negative phi = rolled towards the right (outside of a left turn). rad.
    """
    mass = Parameters["Mass"]
    hPrime = Parameters["CoG-height"] - getRollAxisHeightAtCG()
    totalRollStiffness = getRollStiffness("front") + getRollStiffness("rear") - mass * 9.81 * hPrime
    return -mass * hPrime * accelerationY / totalRollStiffness

def calcAxleLateralLoadTransfer(axleLateralForce:float, rollAngle:float, axle:str) -> float:
    """
    [Pacejka] Eq. 1.35 (p. 20), steady state: dFz_i = (-c_phi_i * phi + Fy_i * h_i) / (2 s_i)
      elastic part:   roll moment carried by this axle's springs/ARB
      geometric part: axle side force acting at its roll centre height h_i
    2 s_i = track width. Returns the load ADDED to the right wheel (and removed from the left), N.
    """
    rollCenterHeight = Parameters[f"{axle}RollCenterHeight"]
    track = Parameters[f"{axle}TrackWidth"]
    return (-getRollStiffness(axle) * rollAngle + axleLateralForce * rollCenterHeight) / track

def calcWheelCamber(rollAngle:float, axle:str) -> np.ndarray:
    """
    [left, right] camber of one axle, rad, car convention (- = top of the wheel leaning in).
    Roll camber (DSS) = camber change of the OUTSIDE wheel per unit body roll, relative to the ground; the
    inside wheel moves the opposite way. Body rolled right (phi < 0) -> right wheel is the outside wheel.
    """
    static = np.radians(Parameters[f"{axle}StaticCamber"])
    change = Parameters[f"{axle}RollCamber"] * rollAngle
    return np.array([static + change, static - change])

def calcWheelInclinations(rollAngle:float) -> np.ndarray:
    """
    [FL, FR, RL, RR] inclination angles in the tyre fit's sign (TTC/SAE IA: + = top of the wheel leaning RIGHT).
    Negative camber leans the left wheel's top right (+IA) and the right wheel's top left (-IA).
    """
    front, rear = calcWheelCamber(rollAngle, "front"), calcWheelCamber(rollAngle, "rear")
    return np.array([-front[0], front[1], -rear[0], rear[1]])

def calcLongitudinalLoadTransfer(accelerationX:float) -> float:
    # Pitch moment balance about the contact patches: dFz = m * a_x * h / L per axle, half per wheel. + a_x unloads the front.
    return Parameters["Mass"] * accelerationX * Parameters["CoG-height"] / Parameters["wheelBase"] / 2.0

def getStaticWheelLoads() -> np.ndarray:
    # [FL, FR, BL, BR] N: b/L of the weight on the front axle, a/L on the rear
    weight = Parameters["Mass"] * 9.81
    a = Parameters["a"]
    b = Parameters["wheelBase"] - a
    front = weight * b / Parameters["wheelBase"] / 2.0
    rear = weight * a / Parameters["wheelBase"] / 2.0
    return np.array([front, front, rear, rear])

def calcLoadTransfer(accelerationX:float, accelerationY:float, frontLateralForce:float, rearLateralForce:float) -> np.ndarray:
    """
    Quasi-static wheel loads [FL, FR, BL, BR], N. Accelerations in the vehicle frame (as an IMU reads them),
    axle lateral forces from the tyre model (previous step).
    """
    rollAngle = calcRollAngle(accelerationY)
    front = calcAxleLateralLoadTransfer(frontLateralForce, rollAngle, "front")
    rear = calcAxleLateralLoadTransfer(rearLateralForce, rollAngle, "rear")
    longitudinal = calcLongitudinalLoadTransfer(accelerationX)

    loads = getStaticWheelLoads() + np.array([
        -longitudinal - front, # FL
        -longitudinal + front, # FR
        longitudinal - rear,   # BL
        longitudinal + rear,   # BR
    ])
    return np.maximum(loads, 0.0) # A tyre can't pull the car down (wheel lift)
