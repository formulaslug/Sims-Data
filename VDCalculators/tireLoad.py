def getLongLoadTransfer(params: dict, accelerationX):
    # TODO: add weight transfer for torsional compliancy
    #frontAxleLoad = params["Mass"] * 9.81 * (params["wheelBase"] - params["frontWeightDist"])/params["wheelBase"] - params["CoG-height"]/params["wheelBase"] * params["Mass"] * accelerationX
    #rearAxleLoad = params["Mass"] * 9.81 * (params["wheelBase"] - params["frontWeightDist"])/params["wheelBase"] - params["CoG-height"]/params["wheelBase"] * params["Mass"] * accelerationX
    #res = [params["Mass"]*9.8/4, params["Mass"]*9.8/4,params["Mass"]*9.8/4,params["Mass"]*9.8/4] # FL, FR, BL, BR
    frontAxleLoad = params["Mass"] * 9.81 * (params["wheelBase"] - params["a"])/params["wheelBase"] - params["CoG-height"]/params["wheelBase"] * params["Mass"] * accelerationX
    rearAxleLoad = params["Mass"] * 9.81 * (params["wheelBase"] - params["a"])/params["wheelBase"] + params["CoG-height"]/params["wheelBase"] * params["Mass"] * accelerationX

    return [frontAxleLoad/2, frontAxleLoad/2, rearAxleLoad/2, rearAxleLoad/2]

def getLatLoadTransfer(params: dict, track, a_y, hcg): # axle track, lateral acceleration, height cg from contact patch (ground)
    if (a_y > 0): #vehicle turning LEFT
        Fn_out = params["Mass"] * ((9.81/2) - a_y*(hcg/track))
        Fn_in = params["Mass"] - Fn_out
    elif (a_y < 0): #vehicle turning RIGHT 
        Fn_out = params["Mass"] * ((9.81/2) + a_y*(hcg/track))
        Fn_in = params["Mass"] - Fn_out
    else:
        return 0
    return (Fn_out, Fn_in)



# -4 -1 1 2
#
