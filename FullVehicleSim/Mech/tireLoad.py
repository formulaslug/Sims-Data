from ..paramLoader import *  # noqa: N999


def calcLoadTransfer(accelerationX:float, accelerationY:float, yawVelocity:float) -> tuple[float, float, float, float]:
    # TODO: add weight transfer for torsional compliancy
    #res = [params["Mass"]*9.8/4, params["Mass"]*9.8/4,params["Mass"]*9.8/4,params["Mass"]*9.8/4] # FL, FR, BL, BR
    frontLeftAxleLoad:float = (Parameters["Mass"] * 9.81 * (Parameters["wheelBase"] - Parameters["a"])/Parameters["wheelBase"] # Static Load
                                - Parameters["CoG-height"] / Parameters["wheelBase"] * Parameters["Mass"] * accelerationX # Longitudinal Load Transfer
                                + Parameters["Cog-height"] / Parameters["frontWheelBase"] * Parameters["Mass"] * accelerationY) # Lateral Load Transfer
    frontRightAxleLoad:float = (Parameters["Mass"] * 9.81 * (Parameters["wheelBase"] - Parameters["a"])/Parameters["wheelBase"] # Static Load
                                - Parameters["CoG-height"]/Parameters["wheelBase"] * Parameters["Mass"] * accelerationX # Longitudinal Load Transfer
                                - Parameters["Cog-height"] / Parameters["frontWheelBase"] * Parameters["Mass"] * accelerationY) # Lateral Load Transfer
    backLeftAxleLoad:float = (Parameters["Mass"] * 9.81 * (Parameters["wheelBase"] - Parameters["a"])/Parameters["wheelBase"] # Static Load
                            + Parameters["CoG-height"]/Parameters["wheelBase"] * Parameters["Mass"] * accelerationX # Longitudinal Load Transfer
                            + Parameters["Cog-height"] / Parameters["frontWheelBase"] * Parameters["Mass"] * accelerationY) # Lateral Load Transfer
    backRightAxleLoad:float = (Parameters["Mass"] * 9.81 * (Parameters["wheelBase"] - Parameters["a"])/Parameters["wheelBase"] # Static Load
                            + Parameters["CoG-height"]/Parameters["wheelBase"] * Parameters["Mass"] * accelerationX # Longitudinal Load Transfer
                            - Parameters["Cog-height"] / Parameters["frontWheelBase"] * Parameters["Mass"] * accelerationY) # Lateral Load Transfer
    return (frontLeftAxleLoad, frontRightAxleLoad, backLeftAxleLoad, backRightAxleLoad) # FL, FR, BL, BR

def calcWeightTransfer():
    # Caleb should do this.
    return 1



# -4 -1 1 2
#
