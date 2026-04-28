import numpy as np

air = np.ones((5, 6)) * 20.0 # Initialize air temperature to 20 C
pack = np.ones((5, 6)) * 20.0 # Initialize pack temperature to 20 C
case = 20.0

ambientTemp = 20.0 # Ambient temperature in C

def conduction(a, b):
    # Simple conduction model: heat transfer proportional to temperature difference
    k = 0.1 # Thermal conductivity
    return k * (a - b)

def convection(a, b):
    # Simple convection model: heat transfer proportional to temperature difference
    h = 0.05 # Convective heat transfer coefficient
    return h * (a - b)

def heat_generation():
    # Simulate heat generation in the battery pack
    return np.random.rand(5, 6) * 5.0 # Random heat generation between 0 and 5 W

for i in range(100):
    conductionRow = conduction(pack[:-1, :], pack[1, :])
    conductionCol = conduction(pack[:, :-1], pack[:, 1:])

    convectionTotal = convection(pack, air)

    heat = heat_generation()

    pack[:-1, :] -= conductionRow
    pack[1:, :] += conductionRow
    pack[:, :-1] -= conductionCol
    pack[:, 1:] += conductionCol

    pack -= convectionTotal
    air += convectionTotal

    pack += heat

    if i % 5 == 0:
        air[1:, :] = air [:-1 , :]
        air[0, :] = np.ones_like(air[0, :]) * ambientTemp