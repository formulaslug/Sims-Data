import numpy as np  # noqa: N999

from ..paramLoader import *


def calcLoadTransfer(accelerationX:float, accelerationY:float, yawVelocity:float) -> np.ndarray:
    """
    Quasi-static wheel loads [FL, FR, BL, BR] in Newtons.

    Accelerations are in the VEHICLE (body) frame, as an IMU would read them:
    x forward, y LEFT (ISO 8855 / Pacejka). Positive accelerationY = turning left,
    which loads the RIGHT (outside) wheels.

    Rigid chassis: lateral transfer is split between axles by their static weight share
    (no roll stiffness distribution yet).
    """
    # TODO: add weight transfer for torsional compliance / roll stiffness distribution
    mass = Parameters["Mass"]
    wheelBase = Parameters["wheelBase"]
    a = Parameters["a"]
    b = wheelBase - a
    h = Parameters["CoG-height"]

    frontStatic = mass * 9.81 * b / wheelBase / 2.0 # per wheel
    rearStatic = mass * 9.81 * a / wheelBase / 2.0  # per wheel

    longTransfer = mass * accelerationX * h / wheelBase / 2.0 # per wheel, + accel moves load rearward
    frontLatTransfer = mass * accelerationY * h / Parameters["frontTrackWidth"] * (b / wheelBase) # + ay moves load to right wheels
    rearLatTransfer = mass * accelerationY * h / Parameters["rearTrackWidth"] * (a / wheelBase)

    loads = np.array([
        frontStatic - longTransfer - frontLatTransfer, # FL
        frontStatic - longTransfer + frontLatTransfer, # FR
        rearStatic + longTransfer - rearLatTransfer,   # BL
        rearStatic + longTransfer + rearLatTransfer,   # BR
    ])
    return np.maximum(loads, 0.0) # A tyre can't pull the car down (wheel lift)

def calcWeightTransfer():
    # Caleb should do this.
    return 1
