from paramLoader import *

def calcLoadTransfer(accelerationX, accelerationY, yawVelocity:np.float64) -> tuple[np.float64, np.float64, np.float64, np.float64]:
    # TODO: add weight transfer for torsional compliancy
    #frontAxleLoad = params["Mass"] * 9.81 * (params["wheelBase"] - params["frontWeightDist"])/params["wheelBase"] - params["CoG-height"]/params["wheelBase"] * params["Mass"] * accelerationX
    #rearAxleLoad = params["Mass"] * 9.81 * (params["wheelBase"] - params["frontWeightDist"])/params["wheelBase"] - params["CoG-height"]/params["wheelBase"] * params["Mass"] * accelerationX
    #res = [params["Mass"]*9.8/4, params["Mass"]*9.8/4,params["Mass"]*9.8/4,params["Mass"]*9.8/4] # FL, FR, BL, BR
    frontAxleLoad:np.float64 = Parameters["Mass"] * 9.81 * (Parameters["wheelBase"] - Parameters["a"])/Parameters["wheelBase"] - Parameters["CoG-height"]/Parameters["wheelBase"] * Parameters["Mass"] * accelerationX
    rearAxleLoad:np.float64 = Parameters["Mass"] * 9.81 * (Parameters["wheelBase"] - Parameters["a"])/Parameters["wheelBase"] + Parameters["CoG-height"]/Parameters["wheelBase"] * Parameters["Mass"] * accelerationX

    return (frontAxleLoad/2, frontAxleLoad/2, rearAxleLoad/2, rearAxleLoad/2) # FL, FR, BL, BR

def calcWeightTransfer():
    # Caleb should do this.
    return 1



# -4 -1 1 2
#
