## Murata VTC 5b cell model (close enough to 5a) Should fix soon though
# Onboarding project: Make a vtc5a model and maybe house them(5a and 5b) both in classes so we can refer to them separately
# https://apps.automeris.io/wpd4/ for getting data from a graph
# https://www.murata.com/-/media/webrenewal/products/batteries/cylindrical/datasheet/us18650vtc5-product-datasheet.ashx?la=en-sg&cvid=20240514015406000000
# Code imported from F# code

import numpy as np
class CellChargeOutOfBounds(Exception):
    pass

class CurrentDrawOutOfBounds(Exception):
    pass

# Voltage tables
def table_half(discharged):
    table = {
        0.00: 4.15,
        0.10: 4.09,
        0.20: 4.06,
        0.30: 4.03,
        0.40: 3.98,
        0.50: 3.94,
        0.60: 3.90,
        0.70: 3.87,
        0.80: 3.84,
        0.90: 3.80,
        1.00: 3.76,
        1.10: 3.72,
        1.20: 3.69,
        1.30: 3.66,
        1.40: 3.63,
        1.50: 3.61,
        1.60: 3.58,
        1.70: 3.56,
        1.80: 3.54,
        1.90: 3.52,
        2.00: 3.50,
        2.10: 3.46,
        2.20: 3.41,
        2.30: 3.35,
        2.40: 3.31,
        2.50: 3.25,
        2.60: 2.99
    }
    if discharged in table:
        return table[discharged]
    else:
        raise CellChargeOutOfBounds(f"Cell Charge out of Bounds: 0.5 A, {discharged} A*H")

def table_5A(discharged):
    table = {
        0.00: 4.05,
        0.10: 3.94,
        0.20: 3.90,
        0.30: 3.86,
        0.40: 3.82,
        0.50: 3.77,
        0.60: 3.73,
        0.70: 3.69,
        0.80: 3.66,
        0.90: 3.62,
        1.00: 3.58,
        1.10: 3.54,
        1.20: 3.50,
        1.30: 3.47,
        1.40: 3.44,
        1.50: 3.41,
        1.60: 3.38,
        1.70: 3.35,
        1.80: 3.32,
        1.90: 3.28,
        2.00: 3.25,
        2.10: 3.20,
        2.20: 3.14,
        2.30: 3.05,
        2.40: 2.86
    }
    if discharged in table:
        return table[discharged]
    else:
        raise CellChargeOutOfBounds(f"Cell Charge out of Bounds: 5 A, {discharged} A*H")

def table_10A(discharged):
    table = {
        0.00: 3.97,
        0.10: 3.81,
        0.20: 3.77,
        0.30: 3.73,
        0.40: 3.68,
        0.50: 3.64,
        0.60: 3.61,
        0.70: 3.57,
        0.80: 3.53,
        0.90: 3.50,
        1.00: 3.46,
        1.10: 3.42,
        1.20: 3.39,
        1.30: 3.35,
        1.40: 3.32,
        1.50: 3.29,
        1.60: 3.26,
        1.70: 3.24,
        1.80: 3.21,
        1.90: 3.18,
        2.00: 3.14,
        2.10: 3.11,
        2.20: 3.05,
        2.30: 3.00,
        2.40: 2.91,
        2.50: 2.80,
        2.60: 1.0
    }
    if discharged in table:
        return table[discharged]
    else:
        raise CellChargeOutOfBounds(f"Cell Charge out of Bounds: 10 A, {discharged} A*H")

def table_15A(discharged):
    table = {
        0.00: 3.83,
        0.10: 3.69,
        0.20: 3.64,
        0.30: 3.60,
        0.40: 3.56,
        0.50: 3.52,
        0.60: 3.49,
        0.70: 3.45,
        0.80: 3.41,
        0.90: 3.38,
        1.00: 3.34,
        1.10: 3.31,
        1.20: 3.28,
        1.30: 3.25,
        1.40: 3.22,
        1.50: 3.19,
        1.60: 3.16,
        1.70: 3.14,
        1.80: 3.11,
        1.90: 3.08,
        2.00: 3.05,
        2.10: 3.01,
        2.20: 2.97,
        2.30: 2.91,
        2.40: 2.83,
        2.50: 2.70,
        2.60: 1.0
    }
    if discharged in table:
        return table[discharged]
    else:
        raise CellChargeOutOfBounds(f"Cell Charge out of Bounds: 15 A, {discharged} A*H")

def table_20A(discharged):
    table = {
        0.00: 3.70,
        0.10: 3.59,
        0.20: 3.54,
        0.30: 3.50,
        0.40: 3.46,
        0.50: 3.43,
        0.60: 3.40,
        0.70: 3.36,
        0.80: 3.33,
        0.90: 3.30,
        1.00: 3.27,
        1.10: 3.25,
        1.20: 3.22,
        1.30: 3.20,
        1.40: 3.17,
        1.50: 3.14,
        1.60: 3.12,
        1.70: 3.10,
        1.80: 3.08,
        1.90: 3.05,
        2.00: 3.02,
        2.10: 2.99,
        2.20: 2.96,
        2.30: 2.92,
        2.40: 2.86,
        2.50: 2.74,
        2.60: 1.0
    }
    if discharged in table:
        return table[discharged]
    else:
        raise CellChargeOutOfBounds(f"Cell Charge out of Bounds: 20 A, {discharged} A*H")

def table_25A(discharged):
    table = {
        0.00: 3.70,
        0.10: 3.49,
        0.20: 3.43,
        0.30: 3.38,
        0.40: 3.34,
        0.50: 3.30,
        0.60: 3.26,
        0.70: 3.22,
        0.80: 3.19,
        0.90: 3.17,
        1.00: 3.15,
        1.10: 3.13,
        1.20: 3.11,
        1.30: 3.10,
        1.40: 3.08,
        1.50: 3.06,
        1.60: 3.04,
        1.70: 3.01,
        1.80: 3.00,
        1.90: 2.97,
        2.00: 2.95,
        2.10: 2.92,
        2.20: 2.88,
        2.30: 2.80,
        2.40: 2.55,
        2.50: 2.20,
        2.60: 1.0
    }
    if discharged in table:
        return table[discharged]
    else:
        raise CellChargeOutOfBounds(f"Cell Charge out of Bounds: 25 A, {discharged} A*H")

def table_30A(discharged):
    table = {
        0.00: 3.56,
        0.10: 3.40,
        0.20: 3.34,
        0.30: 3.28,
        0.40: 3.23,
        0.50: 3.18,
        0.60: 3.14,
        0.70: 3.11,
        0.80: 3.09,
        0.90: 3.07,
        1.00: 3.05,
        1.10: 3.04,
        1.20: 3.04,
        1.30: 3.02,
        1.40: 3.01,
        1.50: 3.01,
        1.60: 2.99,
        1.70: 2.98,
        1.80: 2.96,
        1.90: 2.94,
        2.00: 2.92,
        2.10: 2.88,
        2.20: 2.86,
        2.30: 2.80,
        2.40: 2.70,
        2.50: 2.40,
        2.60: 1.0
    }
    if discharged in table:
        return table[discharged]
    else:
        raise CellChargeOutOfBounds(f"Cell Charge out of Bounds: 30 A, {discharged} A*H")

def table_35A(discharged):
    table = {
        0.00: 3.41,
        0.10: 3.25,
        0.20: 3.20,
        0.30: 3.13,
        0.40: 3.08,
        0.50: 3.03,
        0.60: 3.00,
        0.70: 2.97,
        0.80: 2.95,
        0.90: 2.94,
        1.00: 2.92,
        1.10: 2.92,
        1.20: 2.92,
        1.30: 2.91,
        1.40: 2.91,
        1.50: 2.91,
        1.60: 2.90,
        1.70: 2.88,
        1.80: 2.86,
        1.90: 2.85,
        2.00: 2.75,
        2.10: 2.50,
        2.20: 2.30,
        2.30: 2.00,
        2.40: 1.0,
        2.50: 1.0,
        2.60: 1.0
    }
    if discharged in table:
        return table[discharged]
    else:
        raise CellChargeOutOfBounds(f"Cell Charge out of Bounds: 35 A, {discharged} A*H")

def lookup(charge, current_draw):
    current_per_cell = current_draw / 20.0
    charge_per_cell = charge / 20.0
    discharged = 2.6 - charge_per_cell
    lower = (current_per_cell // 5.0) * 5.0
    upper = ((current_per_cell // 5.0) + 1.0) * 5.0
    if not np.isnan(discharged):
        charge_rounded = round(discharged * 10.0) / 10.0
    else:
        charge_rounded = 0

    def voltage_match(current):
        if current == 0.0:
            return table_half(charge_rounded)
        elif current == 5.0:
            return table_5A(charge_rounded)
        elif current == 10.0:
            return table_10A(charge_rounded)
        elif current == 15.0:
            return table_15A(charge_rounded)
        elif current == 20.0:
            return table_20A(charge_rounded)
        elif current == 25.0:
            return table_25A(charge_rounded)
        elif current == 30.0:
            return table_30A(charge_rounded)
        elif current == 35.0:
            return table_35A(charge_rounded)
        else:
            raise CurrentDrawOutOfBounds(f"Current Draw out of bounds: {current}")

    lower_voltage = voltage_match(lower)
    upper_voltage = voltage_match(upper)
    lower_percent = (current_per_cell - lower) / 5.0
    voltage = lower_percent * upper_voltage + (1.0 - lower_percent) * lower_voltage
    return voltage
