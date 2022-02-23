from typing import Optional, Dict, Callable
import torch

# This file contains various physical constants and functions to convert units
# from the atomic units

__all__ = ["convert_length", "convert_time", "convert_freq", "convert_ir_ints",
           "convert_raman_ints", "convert_edipole", "convert_equadrupole"]

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
    return str(list(converter.keys()))

def _add_docstr_to(phys: str, converter: Dict[str, float]) -> Callable:
    # automatically add docstring for converter functions

    def decorator(callable: Callable):
        callable.__doc__ = f"""
            Convert the {phys} from a unit to another unit.
            Available units are (case-insensitive): ``{_avail_keys(converter)}``

            Arguments
            ---------
            a: torch.Tensor
                The tensor to be converter.
            from_unit: str or None
                The unit of ``a``. If ``None``, it is assumed to be in atomic unit.
            to_unit: str or None
                The unit for ``a`` to be converted to. If ``None``, it is assumed
                to be converted to the atomic unit.

            Returns
            -------
            torch.Tensor
                The tensor in the new unit.
        """
        return callable
    return decorator

@_add_docstr_to("time", _time_converter)
def convert_time(a: PhysVarType, from_unit: UnitType = None,
                 to_unit: UnitType = None) -> PhysVarType:
    # convert unit time from atomic unit to the given unit
    return _converter(a, from_unit, to_unit, _time_converter)

@_add_docstr_to("frequency", _freq_converter)
def convert_freq(a: PhysVarType, from_unit: UnitType = None,
                 to_unit: UnitType = None) -> PhysVarType:
    # convert unit frequency from atomic unit to the given unit
    return _converter(a, from_unit, to_unit, _freq_converter)

@_add_docstr_to("IR intensity", _ir_ints_converter)
def convert_ir_ints(a: PhysVarType, from_unit: UnitType = None,
                    to_unit: UnitType = None) -> PhysVarType:
    # convert unit IR intensity from atomic unit to the given unit
    return _converter(a, from_unit, to_unit, _ir_ints_converter)

@_add_docstr_to("Raman intensity", _raman_ints_converter)
def convert_raman_ints(a: PhysVarType, from_unit: UnitType = None,
                       to_unit: UnitType = None) -> PhysVarType:
    # convert unit IR intensity from atomic unit to the given unit
    return _converter(a, from_unit, to_unit, _raman_ints_converter)

@_add_docstr_to("length", _length_converter)
def convert_length(a: PhysVarType, from_unit: UnitType = None,
                   to_unit: UnitType = None) -> PhysVarType:
    # convert unit length from atomic unit to the given unit
    return _converter(a, from_unit, to_unit, _length_converter)

@_add_docstr_to("electric dipole", _edipole_converter)
def convert_edipole(a: PhysVarType, from_unit: UnitType = None,
                    to_unit: UnitType = None) -> PhysVarType:
    # convert unit electric dipole from atomic unit to the given unit
    return _converter(a, from_unit, to_unit, _edipole_converter)

@_add_docstr_to("electric quadrupole", _equadrupole_converter)
def convert_equadrupole(a: PhysVarType, from_unit: UnitType = None,
                        to_unit: UnitType = None) -> PhysVarType:
    # convert unit electric dipole from atomic unit to the given unit
    return _converter(a, from_unit, to_unit, _equadrupole_converter)

def _converter(a: PhysVarType, from_unit: UnitType, to_unit: UnitType,
               converter: Dict[str, float]) -> PhysVarType:
    # converter from a unit to another unit
    from_unit = _preproc_unit(from_unit)
    to_unit = _preproc_unit(to_unit)
    if from_unit == to_unit:
        return a
    if from_unit is not None:
        a = a / _get_converter_value(converter, from_unit)
    if to_unit is not None:
        a = a * _get_converter_value(converter, to_unit)
    return a

def _get_converter_value(converter: Dict[str, float], unit: UnitType) -> float:
    if unit not in converter:
        avail_units = _avail_keys(converter)
        raise ValueError(f"Unknown unit: {unit}. Available units are: {avail_units}")
    return converter[unit]

def _preproc_unit(unit: UnitType):
    if unit is None:
        return unit
    else:
        return ''.join(unit.lower().split())
