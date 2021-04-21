from typing import Optional, Dict, Callable
import torch

# This file contains various physical constants and functions to convert units
# from the atomic units

__all__ = ["length_to", "time_to", "freq_to", "ir_ints_to", "raman_ints_to",
           "edipole_to", "equadrupole_to"]

# 1 atomic unit in SI
LENGTH = 5.29177210903e-11  # m
TIME = 2.4188843265857e-17  # s
CHARGE = 1.602176634e-19  # C

# 1 atomic unit in other unit
DEBYE = 2.541746473  # Debye (for dipole)
ANGSTROM = LENGTH / 1e-10  # angstrom (length)
AMU = 5.485799090649e-4  # atomic mass unit (mass)

# constants in SI
LIGHT_SPEED = 2.99792458e8  # m/s

# scales
ATTO = 1e-15
FEMTO = 1e-12
NANO = 1e-9
MICRO = 1e-6
MILLI = 1e-3
CENTI = 1e-2
DECI = 1e-1
KILO = 1e3
MEGA = 1e6
GIGA = 1e9
TERA = 1e12

PhysVarType = torch.Tensor
UnitType = Optional[str]

_length_converter = {
    "angst": ANGSTROM,
    "angstrom": ANGSTROM,
    "m": LENGTH,
    "cm": LENGTH / CENTI,
}

_freq_converter = {
    "cm-1": CENTI / TIME / LIGHT_SPEED,
    "cm^-1": CENTI / TIME / LIGHT_SPEED,
    "hz": 1.0 / TIME,
    "khz": 1.0 / TIME / KILO,
    "mhz": 1.0 / TIME / MEGA,
    "ghz": 1.0 / TIME / GIGA,
    "thz": 1.0 / TIME / TERA,
}

_ir_ints_converter = {
    "(debye/angst)^2/amu": (DEBYE / ANGSTROM) ** 2 / AMU,
    "km/mol": (DEBYE / ANGSTROM) ** 2 / AMU * 42.256,  # from https://dx.doi.org/10.1002%2Fjcc.24344
}

_raman_ints_converter = {
    "angst^4/amu": ANGSTROM ** 4 / AMU,
}

_time_converter = {
    "s": TIME,
    "us": TIME / MICRO,
    "ns": TIME / NANO,
    "fs": TIME / FEMTO,
}

_edipole_converter = {
    "d": DEBYE,
    "debye": DEBYE,
    "c*m": DEBYE,  # Coulomb meter
}

_equadrupole_converter = {
    "debye*angst": DEBYE * ANGSTROM  # Debye angstrom
}

def _avail_keys(converter: Dict[str, float]) -> str:
    # returns the available keys in a string of list of string
    return str(list(_length_converter.keys()))

def _add_docstr_to(phys: str, converter: Dict[str, float]) -> Callable:
    # automatically add docstring for converter functions

    def decorator(callable: Callable):
        callable.__doc__ = f"""
            Convert the {phys} from atomic unit to the given unit.
            Available units are (case-insensitive): {_avail_keys(converter)}
        """
        return callable
    return decorator

@_add_docstr_to("time", _time_converter)
def time_to(a: PhysVarType, unit: UnitType) -> PhysVarType:
    # convert unit time from atomic unit to the given unit
    return _converter_to(a, unit, _time_converter)

@_add_docstr_to("frequency", _freq_converter)
def freq_to(a: PhysVarType, unit: UnitType) -> PhysVarType:
    # convert unit frequency from atomic unit to the given unit
    return _converter_to(a, unit, _freq_converter)

@_add_docstr_to("IR intensity", _ir_ints_converter)
def ir_ints_to(a: PhysVarType, unit: UnitType) -> PhysVarType:
    # convert unit IR intensity from atomic unit to the given unit
    return _converter_to(a, unit, _ir_ints_converter)

@_add_docstr_to("Raman intensity", _raman_ints_converter)
def raman_ints_to(a: PhysVarType, unit: UnitType) -> PhysVarType:
    # convert unit IR intensity from atomic unit to the given unit
    return _converter_to(a, unit, _raman_ints_converter)

@_add_docstr_to("length", _length_converter)
def length_to(a: PhysVarType, unit: UnitType) -> PhysVarType:
    # convert unit length from atomic unit to the given unit
    return _converter_to(a, unit, _length_converter)

@_add_docstr_to("electric dipole", _edipole_converter)
def edipole_to(a: PhysVarType, unit: UnitType) -> PhysVarType:
    # convert unit electric dipole from atomic unit to the given unit
    return _converter_to(a, unit, _edipole_converter)

@_add_docstr_to("electric quadrupole", _equadrupole_converter)
def equadrupole_to(a: PhysVarType, unit: UnitType) -> PhysVarType:
    # convert unit electric dipole from atomic unit to the given unit
    return _converter_to(a, unit, _equadrupole_converter)

def _converter_to(a: PhysVarType, unit: UnitType, converter: Dict[str, float]) -> PhysVarType:
    # converter from the atomic unit
    if unit is None:
        return a
    u = unit.lower()
    try:
        return a * converter[u]
    except KeyError:
        avail_units = _avail_keys(converter)
        raise ValueError(f"Unknown unit: {unit}. Available units are: {avail_units}")
