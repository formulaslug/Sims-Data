import numpy as np
from paramLoader import *
import matplotlib.pyplot as plt

np.random.seed(42)

class Cell:
    def __init__ (self, soc, resistance, capacity_Ah = 5, hysteresis = 0):
        self.soc = soc
        self.resistance = resistance
        self.capacity_Ah = capacity_Ah
        self.hysteresis = hysteresis

    def ocv(self, T_K = 298.15):
        #code from powertrain
        F = Parameters["FaradaysConstant"]
        R = Parameters["GasConstant"]

        V0 = Magic["cellModel_V0"]
        C1 = Magic["cellModel_C1"]
        C2 = Magic["cellModel_C2"]
        C3 = Magic["cellModel_C3"]
        C4 = Magic["cellModel_C4"]

        eps = 1e-9

        soc_shift = self.soc - (0.1 ** 3)

        denom = np.clip(1.0 - soc_shift + C4, eps, None)
        numer = np.clip(C1 * soc_shift + C3, eps, None)

        log_term = np.log(numer / denom)
        return V0 + (C2 * (R * T_K / F) * log_term) + Magic["cellModel_bias"]

    def terminal_voltage(self, current):
        #main function
        return (self.ocv() - self.hysteresis - (current * self.resistance))
    
    def update_soc(self, current, dt, capacity_Ah):
        #change w time
        change_soc = (current * dt / 3600) / capacity_Ah
        self.soc = np.clip(self.soc - change_soc, 0.0, 1.0)
    
class Parallel_Groups:
    def __init__(self, cells):
        self.cells = cells

    def solve_currents(self, pack_current):
        sum_rhs_over_R = 0  ##rhs is ocv - hysteris (right hand side of equation)
        sum_1_over_R = 0

        for cell in self.cells:
            rhs = cell.ocv() - cell.hysteresis
            sum_rhs_over_R += rhs/cell.resistance
            sum_1_over_R += 1/cell.resistance

        V_group = (sum_rhs_over_R - pack_current)/sum_1_over_R
        currents = []
        for cell in self.cells:
            rhs = cell.ocv() - cell.hysteresis
            current = (rhs - V_group)/cell.resistance
            currents.append(float(current))
        
        return currents, V_group

    def step(self, pack_current, dt, capacity_Ah):
        #for each timestep
        currents, V_group = self.solve_currents(pack_current)
        for cell, current in zip(self.cells, currents):
            cell.update_soc(current, dt, capacity_Ah)
        return currents, V_group

class Module:
    def __init__(self, groups):
        self.groups = groups
    
    def step(self, pack_current, dt, capacity_Ah):
        total_voltage = 0
        all_currents = []
        for group in self.groups:
            currents, V_group = group.step(pack_current, dt, capacity_Ah)
            total_voltage += V_group
            all_currents.append(currents)
        return all_currents, total_voltage

def build_modules(width = 4, height = 2, resistances = None):
    #if no given csv
    if resistances is None:
        resistances = np.random.normal(0.008, 0.0005, (height, width))
    #with importing a csv
    else:
        resistances = np.asarray(resistances)
        if resistances.shape != (height, width):
            raise ValueError(f"Expected shape ({height}, {width}), got {resistances.shape}")
    groups = []
    for h in range(height):
        cells = []
        for w in range(width):
            soc = np.random.uniform(0.85, 0.95)
            cells.append(Cell(soc=soc, resistance=resistances[h, w]))
        groups.append(Parallel_Groups(cells))
    return Module(groups)    

module = build_modules(width=4, height=2)   
##sim

pack_current = 100.0
dt = 1.0
capacity_Ah = 5.0

num_steps = 10

for step in range(num_steps):
    all_currents, module_voltage = module.step(pack_current, dt, capacity_Ah)
    print(f"\nStep {step}")
    print(f"Module Voltage: {module_voltage:.2f} V")
    
    for g, group_currents in enumerate(all_currents):
        print(f"  Group {g}: {group_currents}")
        print(f"  Group {g} Sum: {np.sum(group_currents):.2f} A")