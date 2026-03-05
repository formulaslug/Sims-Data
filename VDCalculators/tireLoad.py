def getloadTransfer(params: dict, accelerationX, accelerationY, yawVelocity:float):
    # TODO: add weight transfer for torsional compliancy
    #frontAxleLoad = params["Mass"] * 9.81 * (params["wheelBase"] - params["frontWeightDist"])/params["wheelBase"] - params["CoG-height"]/params["wheelBase"] * params["Mass"] * accelerationX
    #rearAxleLoad = params["Mass"] * 9.81 * (params["wheelBase"] - params["frontWeightDist"])/params["wheelBase"] - params["CoG-height"]/params["wheelBase"] * params["Mass"] * accelerationX
    #res = [params["Mass"]*9.8/4, params["Mass"]*9.8/4,params["Mass"]*9.8/4,params["Mass"]*9.8/4] # FL, FR, BL, BR
    frontAxleLoad = params["Mass"] * 9.81 * (params["wheelBase"] - params["a"])/params["wheelBase"] - params["CoG-height"]/params["wheelBase"] * params["Mass"] * accelerationX
    rearAxleLoad = params["Mass"] * 9.81 * (params["wheelBase"] - params["a"])/params["wheelBase"] + params["CoG-height"]/params["wheelBase"] * params["Mass"] * accelerationX

    return [frontAxleLoad/2, frontAxleLoad/2, rearAxleLoad/2, rearAxleLoad/2]

def getWeightTransfer(params:dict):
    # Caleb should do this.
    return 1



# -4 -1 1 2
#
