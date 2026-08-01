import numpy as np

air = np.ones((5, 6)) * 20.0 # Initialize air temperature to 20 C
pack = np.ones((5, 6)) * 20.0 # Initialize pack temperature to 20 C
case = 20.0

ambientTemp = 20.0 # Ambient temperature in C

def conduction(a, b, k):
    # Simple conduction model: heat transfer proportional to temperature difference
    return k * (a - b)

def convection(a, b):
    # Simple convection model: heat transfer proportional to temperature difference
    h = 0.05 # Convective heat transfer coefficient
    return h * (a - b)

def heat_generation():
    # Simulate heat generation in the battery pack
    return np.random.rand(5, 6) * 1.0 # Random heat generation between 0 and 5 W

airS = np.zeros((100, 5, 6))
packS = np.zeros((100, 5, 6))
caseTemps = np.zeros(100)

for i in range(100):
    conductionRow = conduction(pack[:-1, :], pack[1, :], 0.1)
    conductionCol = conduction(pack[:, :-1], pack[:, 1:], 0.1)

    topCaseConduction = conduction(pack[0, :], case, 0.01)
    bottomCaseConduction = conduction(pack[-1, :], case, 0.01)
    leftConduction = conduction(pack[1:-1, 0], case, 0.01)
    rightConduction = conduction(pack[1:-1, -1], case, 0.01)

    convectionTotal = convection(pack, air)

    case += np.sum([np.sum(topCaseConduction), np.sum(bottomCaseConduction), np.sum(leftConduction), np.sum(rightConduction)])

    heat = heat_generation()

    pack[:-1, :] -= conductionRow
    pack[1:, :] += conductionRow
    pack[:, :-1] -= conductionCol
    pack[:, 1:] += conductionCol

    pack[0, :] -= topCaseConduction
    pack[-1, :] -= bottomCaseConduction
    pack[1:-1, 0] -= leftConduction
    pack[1:-1, -1] -= rightConduction

    pack -= convectionTotal
    air += convectionTotal

    pack += heat

    if i % 5 == 0:
        air[1:, :] = air [:-1 , :]
        air[0, :] = np.ones_like(air[0, :]) * ambientTemp