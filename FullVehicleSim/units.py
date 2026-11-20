"""
units.py

Unit setup for FullVehicleSim.

- CHECK_UNITS = True : include units and check them (debugging / testing).
- CHECK_UNITS = False: plain floats in base SI units (full speed).
"""

import numpy as np
import unyt as u
from paramLoader import *

CHECK_UNITS = True   # True: values include units and are checked (debugging). False: plain floats (full speed)


def unit(value, units):
    # Attaches units to a value, or returns plain SI numbers when CHECK_UNITS is off.
    quantity = value * u.Unit(units)
    if CHECK_UNITS:
        return quantity
    si = quantity.in_mks().v
    return float(si) if np.ndim(si) == 0 else si


def with_units(value, units):
    # For printing/plotting: includes units in either mode (attaches units only if missing).
    return value if isinstance(value, u.unyt_array) else value * u.Unit(units)


RAD = u.rad if CHECK_UNITS else 1.0   # used to add or remove radians


# Parameters: the unit each value in params.json5 is written in
PARAM_UNITS = {
    "stepsPerSecond":     "1/s",
    "ambientTemperature": "degC",
    "airDensity":         "kg/m**3",
    "dragCoeffAreaCombo": "m**2",       # drag coefficient * frontal area
    "Mass":               "kg",
    "brakeCaliperArea":   "inch**2",    # caliper piston area
    "brakeDiscRadius":    "m",
    "wheelRadius":        "m",
}

# P["name"] is a parameter with units (or a plain SI float in fast mode)
P = {name: unit(Parameters[name], units) for name, units in PARAM_UNITS.items()}


# worldArray columns: the unit each column is stored in
# Most are SI. Brake pressures (psi) and brake temperatures (degC) are not, so
# get() and to_column() convert them to/from SI in fast mode.
COLUMN_UNITS = {
    varSpeed: "m/s",
    varDrag:  "N",
    varFrontBrakeTemperature: "degC",
    varRearBrakeTemperature:  "degC",
    varBrakePressureFront:    "psi",
    varBrakePressureRear:     "psi",
    varFrontBrakeForce:       "N",
    varRearBrakeForce:        "N",
}
COLUMN_UNITS = {var: u.Unit(units) for var, units in COLUMN_UNITS.items()}

# Fast mode conversion for each column: SI value = stored * scale + offset
_SCALE  = {var: float((1 * U).in_mks().v - (0 * U).in_mks().v) for var, U in COLUMN_UNITS.items()}
_OFFSET = {var: float((0 * U).in_mks().v) for var, U in COLUMN_UNITS.items()}


if CHECK_UNITS:
    def get(worldArray, step, var):
        """Reads a stored worldArray value and attaches its column's unit."""
        return u.unyt_quantity(float(worldArray[step, var]), COLUMN_UNITS[var])
    def to_column(value, var):
        """Converts a result to its column's unit (error if it can't) and returns the plain number."""
        return value.to_value(COLUMN_UNITS[var])
else:
    def get(worldArray, step, var):
        return worldArray[step, var] * _SCALE[var] + _OFFSET[var]

    def to_column(value, var):
        return (value - _OFFSET[var]) / _SCALE[var]

# Fast mode conversion from SI to any unit, calculated once per unit and reused
_UNIT_CONVERSIONS = {}

def to_unit(value, units):
    """Returns a value as a plain number in the given units.
    Checked mode: converts, with an error if the value is the wrong kind of quantity.
    Fast mode: the value is plain SI, so it is converted with a scale and offset."""
    if CHECK_UNITS:
        return value.to_value(units)
    if units not in _UNIT_CONVERSIONS:
        U = u.Unit(units)
        offset = float((0 * U).in_mks().v)
        _UNIT_CONVERSIONS[units] = (float((1 * U).in_mks().v) - offset, offset)
    scale, offset = _UNIT_CONVERSIONS[units]
    return (value - offset) / scale