from paramLoader import Parameters

def calcLoadTransfer(accelerationX, accelerationY, yawVelocity:float):
    # TODO: add weight transfer for torsional compliancy
    #frontAxleLoad = params["Mass"] * 9.81 * (params["wheelBase"] - params["frontWeightDist"])/params["wheelBase"] - params["CoG-height"]/params["wheelBase"] * params["Mass"] * accelerationX
    #rearAxleLoad = params["Mass"] * 9.81 * (params["wheelBase"] - params["frontWeightDist"])/params["wheelBase"] - params["CoG-height"]/params["wheelBase"] * params["Mass"] * accelerationX
    #res = [params["Mass"]*9.8/4, params["Mass"]*9.8/4,params["Mass"]*9.8/4,params["Mass"]*9.8/4] # FL, FR, BL, BR
    frontAxleLoad = Parameters["Mass"] * 9.81 * (Parameters["wheelBase"] - Parameters["a"])/Parameters["wheelBase"] - Parameters["CoG-height"]/Parameters["wheelBase"] * Parameters["Mass"] * accelerationX
    rearAxleLoad = Parameters["Mass"] * 9.81 * (Parameters["wheelBase"] - Parameters["a"])/Parameters["wheelBase"] + Parameters["CoG-height"]/Parameters["wheelBase"] * Parameters["Mass"] * accelerationX

    return [frontAxleLoad/2, frontAxleLoad/2, rearAxleLoad/2, rearAxleLoad/2]

def calcWeightTransfer():
    # Caleb should do this.
    return 1



# -4 -1 1 2
#
