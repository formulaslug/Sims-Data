def getLongLoadTransfer(params: dict, accelerationX):
    # TODO: add weight transfer for torsional compliancy
    #frontAxleLoad = params["Mass"] * 9.81 * (params["wheelBase"] - params["frontWeightDist"])/params["wheelBase"] - params["CoG-height"]/params["wheelBase"] * params["Mass"] * accelerationX
    #rearAxleLoad = params["Mass"] * 9.81 * (params["wheelBase"] - params["frontWeightDist"])/params["wheelBase"] - params["CoG-height"]/params["wheelBase"] * params["Mass"] * accelerationX
    #res = [params["Mass"]*9.8/4, params["Mass"]*9.8/4,params["Mass"]*9.8/4,params["Mass"]*9.8/4] # FL, FR, BL, BR
    frontAxleLoad = params["Mass"] * 9.81 * (params["wheelBase"] - params["a"])/params["wheelBase"] - params["CoG-height"]/params["wheelBase"] * params["Mass"] * accelerationX
    rearAxleLoad = params["Mass"] * 9.81 * (params["wheelBase"] - params["a"])/params["wheelBase"] + params["CoG-height"]/params["wheelBase"] * params["Mass"] * accelerationX

    return [frontAxleLoad/2, frontAxleLoad/2, rearAxleLoad/2, rearAxleLoad/2]

def getLatLoadTransfer(params: dict, track, a_y, hcg): # axle track, lateral acceleration, height cg from contact patch (ground)
    mass_axle   = params["Mass"]
    static_load = mass_axle * 9.81 / 2          # per-wheel static load on the axle
    latTransfer = mass_axle * abs(a_y) * hcg / track
    Fn_outside = static_load - latTransfer
    Fn_inside = static_load + latTransfer

    return Fn_outside, Fn_inside



# -4 -1 1 2
#
