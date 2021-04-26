from collections import defaultdict
from typing import Union, List, Optional, Mapping, Callable
import torch
from dqc.grid.base_grid import BaseGrid
from dqc.grid.radial_grid import RadialGrid, LogM3Transformation, \
                                 TreutlerM4Transformation, DE2Transformation
from dqc.grid.lebedev_grid import LebedevGrid, TruncatedLebedevGrid
from dqc.grid.multiatoms_grid import BeckeGrid, PBCBeckeGrid
from dqc.grid.truncation_rules import DasguptaTrunc, NoTrunc
from dqc.hamilton.intor.lattice import Lattice
from dqc.utils.periodictable import atom_bragg_radii, atom_expected_radii
from dqc.utils.misc import get_option

__all__ = ["get_grid", "get_predefined_grid"]

# list of alphas for de2 transformation from https://onlinelibrary.wiley.com/doi/epdf/10.1002/jcc.24761
__sg2_dasgupta_alphas = defaultdict(lambda: 1.0, {
    1: 2.6,
    3: 3.2,
    4: 2.4,
    5: 2.4,
    6: 2.2,
    7: 2.2,
    8: 2.2,
    9: 2.2,
    11: 3.2,
    12: 2.4,
    13: 2.5,
    14: 2.3,
    15: 2.5,
    16: 2.5,
    17: 2.5,
})
__sg3_dasgupta_alphas = defaultdict(lambda: 1.0, {
    1: 2.7,
    3: 3.0,
    4: 2.4,
    5: 2.4,
    6: 2.4,
    7: 2.4,
    8: 2.6,
    9: 2.1,
    11: 3.2,
    12: 2.6,
    13: 2.6,
    14: 2.8,
    15: 2.4,
    16: 2.4,
    17: 2.6,
})

# list of optimized xi for M4 transformation from Treutler Table I
# https://doi.org/10.1063/1.469408
__treutler_xi = defaultdict(lambda: 1.0, {
    1: 0.8,
    2: 0.9,
    3: 1.8,
    4: 1.4,
    5: 1.3,
    6: 1.1,
    7: 0.9,
    8: 0.9,
    9: 0.9,
    10: 0.9,
    11: 1.4,
    12: 1.3,
    13: 1.3,
    14: 1.2,
    15: 1.1,
    16: 1.0,
    17: 1.0,
    18: 1.0,
    19: 1.5,
    20: 1.4,
    21: 1.3,
    22: 1.2,
    23: 1.2,
    24: 1.2,
    25: 1.2,
    26: 1.2,
    27: 1.2,
    28: 1.1,
    29: 1.1,
    30: 1.1,
    31: 1.1,
    32: 1.0,
    33: 0.9,
    34: 0.9,
    35: 0.9,
    36: 0.9,
})

# number of angular points to precision
__nang2prec = {
    6: 3,
    14: 5,
    26: 7,
    38: 9,
    50: 11,
    74: 13,
    86: 15,
    110: 17,
    146: 19,
    170: 21,
    194: 23,
    230: 25,
    266: 27,
    302: 29,
    350: 31,
    434: 35,
    590: 41,
    770: 47,
    974: 53,
    1202: 59,
    1454: 65,
    1730: 71,
    2030: 77,
    2354: 83,
    2702: 89,
    3074: 95,
    3470: 101,
    3890: 107,
    4334: 113,
    4802: 119,
    5294: 125,
    5810: 131,
}

_dtype = torch.double
_device = torch.device("cpu")

def get_grid(atomzs: Union[List[int], torch.Tensor], atompos: torch.Tensor,
             *,
             lattice: Optional[Lattice] = None,
             nr: int = 99,
             nang: int = 590,
             radgrid_generator: str = "uniform",
             radgrid_transform: str = "sg2-dasgupta",
             atom_radii: str = "expected",
             multiatoms_scheme: str = "becke",
             truncate: Optional[str] = "dasgupta",
             dtype: torch.dtype = _dtype,
             device: torch.device = _device) -> BaseGrid:
    # atompos: (natoms, ndim)
    assert atompos.ndim == 2
    assert atompos.shape[-2] == len(atomzs)

    # convert the atomzs to a list of integers
    if isinstance(atomzs, torch.Tensor):
        assert atomzs.ndim == 1
        atomzs_list = [a.item() for a in atomzs]
    else:
        atomzs_list = list(atomzs)

    # get the atom radii list
    atom_radii_options: Mapping[str, Union[List[float]]] = {
        "expected": atom_expected_radii,
        "bragg": atom_bragg_radii,
    }
    atom_radii_list = get_option("atom radii", atom_radii, atom_radii_options)
    atomradii = torch.tensor([atom_radii_list[atomz] for atomz in atomzs_list],
                             dtype=dtype, device=device)

    # construct the radial grid transformation as a function of atom z
    radgrid_tf_options = {
        "sg2-dasgupta":
            lambda atz: DE2Transformation(
                alpha=__sg2_dasgupta_alphas[atz], rmin=1e-7, rmax=15 * atom_radii_list[atz]),
        "sg3-dasgupta":
            lambda atz: DE2Transformation(
                alpha=__sg3_dasgupta_alphas[atz], rmin=1e-7, rmax=15 * atom_radii_list[atz]),
        "logm3":
            lambda atz: LogM3Transformation(ra=atom_radii_list[atz]),
        "treutlerm4":
            lambda atz: TreutlerM4Transformation(xi=__treutler_xi[atz], alpha=0.6),
    }
    radgrid_tf = get_option("radial grid transformation", radgrid_transform, radgrid_tf_options)

    # get the truncation rule as a function to avoid unnecessary evaluation
    trunc_options = {
        "dasgupta": lambda: DasguptaTrunc(nr),
        "no": lambda: NoTrunc(),
    }
    truncate_str = truncate if truncate is not None else "no"
    trunc = get_option("truncation rule", truncate_str, trunc_options)()

    prec = get_option("number of angular points", nang, __nang2prec)
    sphgrids: List[BaseGrid] = []
    for (atz, atpos) in zip(atomzs_list, atompos):
        radgrid = RadialGrid(nr, grid_integrator=radgrid_generator,
                             grid_transform=radgrid_tf(atz), dtype=dtype, device=device)
        if trunc.to_truncate(atz):
            rad_slices = trunc.rad_slices(atz)
            radgrids: List[BaseGrid] = [radgrid[sl] for sl in rad_slices]
            precs = trunc.precs(atz)
            sphgrid = TruncatedLebedevGrid(radgrids, precs)
        else:
            sphgrid = LebedevGrid(radgrid, prec=prec)
        sphgrids.append(sphgrid)

    # get the multi atoms grid
    # the values are a function to avoid constructing it unnecessarily
    if lattice is None:
        multiatoms_options: Mapping[str, Callable[[], BaseGrid]] = {
            "becke": lambda: BeckeGrid(sphgrids, atompos, atomradii=atomradii),
        }
    else:
        assert isinstance(lattice, Lattice)
        multiatoms_options = {
            "becke": lambda: PBCBeckeGrid(sphgrids, atompos, lattice=lattice),  # type: ignore
        }
    grid = get_option("multiatoms scheme", multiatoms_scheme, multiatoms_options)()
    return grid

def get_predefined_grid(grid_inp: Union[int, str], atomzs: Union[List[int], torch.Tensor],
                        atompos: torch.Tensor,
                        *,
                        lattice: Optional[Lattice] = None,
                        dtype: torch.dtype = _dtype, device: torch.device = _device) -> BaseGrid:
    """
    Returns the predefined grid object given the grid name.
    """
    if isinstance(grid_inp, str):
        if grid_inp == "sg2":
            return get_grid(atomzs, atompos, lattice=lattice,
                            nr=75, nang=302,
                            radgrid_generator="uniform",
                            radgrid_transform="sg2-dasgupta",
                            atom_radii="expected",
                            multiatoms_scheme="becke",
                            truncate="dasgupta",
                            dtype=dtype, device=device)
        elif grid_inp == "sg3":
            return get_grid(atomzs, atompos, lattice=lattice,
                            nr=99, nang=590,
                            radgrid_generator="uniform",
                            radgrid_transform="sg3-dasgupta",
                            atom_radii="expected",
                            multiatoms_scheme="becke",
                            truncate="dasgupta",
                            dtype=dtype, device=device)
        else:
            raise ValueError(f"Unknown grid name: {grid_inp}")
    elif isinstance(grid_inp, int):
        # grid_inp as an int is deprecated (TODO: put a warning here)
        #        0,   1,   2,   3,   4,    5
        nr   = [20,  40,  60,  75,  99,  125][grid_inp]
        nang = [74, 110, 170, 302, 590, 1202][grid_inp]
        return get_grid(atomzs, atompos, lattice=lattice,
                        nr=nr, nang=nang,
                        radgrid_generator="chebyshev",
                        radgrid_transform="treutlerm4",
                        atom_radii="bragg",
                        multiatoms_scheme="becke",
                        truncate=None,
                        dtype=dtype, device=device)
    else:
        raise TypeError("Unknown type of grid_inp: %s" % type(grid_inp))
