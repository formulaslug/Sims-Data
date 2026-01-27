from paramLoader import Parameters, Magic
from state import VehicleState
from Mech.braking import calcBrakeForce
from Mech.aero import calcDrag

def calcResistiveForces(worldPrev:VehicleState, inputs):
        if worldPrev.speed <= 1e-5: # Floating point error
            return 0
        else:
            brakeForce = calcBrakeForce(worldPrev, inputs)
            return -1 * (calcDrag(worldPrev) + brakeForce)
        
