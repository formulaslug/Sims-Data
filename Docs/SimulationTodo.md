# Simulation Todo

1. Better drag model that takes into account aeropackage (FS-3)
1. Individual wheel models
    1. Suspension
        1. Travel (x)
        1. Velocity (v)
        1. How will this react under acceleration in any direction (steering causes lateral acceleration) (throttle/brakes causes longitudinal acceleration)
    1. Wheel
        1. Brake temp (more complex soon)
        1. Wheel temp (more complex soon)
        1. Wheel rpm/speed
1. Differential + Drivetrain losses
    1. Energy loss due to chain, tripods, axle, hub/upright rubbing
    1. Model rolling resistance better
    1. Model limited slip differential
    1. Log losses so we have an idea of energy loss in the drivetrain
1. Motor Efficiency + Heating
    1. Function of temp and current draw
    1. Keep track of losses
    1. Efficiency loss goes into heat of motor. Need an estimate of its thermal mass and then change in temp. (Motor temp new var)
1. Cleaner logging
    1. Log everything without having to add more rows constantly
    1. Keep efficiency in mind
1. Tractive system heat generation (Not acc)
    1. Estimate how much heat is generated in the accumulator
    1. Not high priority unless we can get more data.
