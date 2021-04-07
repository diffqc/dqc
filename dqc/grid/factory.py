from typing import Union, List, Optional
import torch
from dqc.grid.base_grid import BaseGrid
from dqc.grid.radial_grid import RadialGrid
from dqc.grid.lebedev_grid import LebedevGrid
from dqc.grid.becke_grid import BeckeGrid, PBCBeckeGrid
from dqc.grid.predefined_grid import SG2, SG3
from dqc.hamilton.intor.lattice import Lattice

__all__ = ["get_grid", "get_atomic_grid"]

GRID_CLS_MAP = {
    "sg2": SG2,
    "sg3": SG3,
}

_dtype = torch.double
_device = torch.device("cpu")

def get_grid(grid_inp: Union[int, str], atomzs: Union[List[int], torch.Tensor], atompos: torch.Tensor,
             *,
             lattice: Optional[Lattice] = None,
             dtype: torch.dtype = _dtype, device: torch.device = _device) -> BaseGrid:
    """
    Returns the grid object given the grid input description.
    """
    # atompos: (natoms, ndim)
    assert atompos.ndim == 2
    assert atompos.shape[-2] == len(atomzs)

    # convert the atomzs to a list of integers
    if isinstance(atomzs, torch.Tensor):
        assert atomzs.ndim == 1
        atomzs_list = [a.item() for a in atomzs]
    else:
        atomzs_list = list(atomzs)

    if isinstance(grid_inp, int) or isinstance(grid_inp, str):
        sphgrids = [get_atomic_grid(grid_inp, atomz, dtype=dtype, device=device)
                    for atomz in atomzs_list]
        if lattice is None:
            return BeckeGrid(sphgrids, atompos)
        else:
            return PBCBeckeGrid(sphgrids, atompos, lattice=lattice)
    else:
        raise TypeError("Unknown type of grid_inp: %s" % type(grid_inp))

def get_atomic_grid(grid_inp: Union[int, str], atomz: int,
                    dtype: torch.dtype = _dtype,
                    device: torch.device = _device) -> BaseGrid:
    """
    Returns an individual atomic grid centered at (0, 0, 0) given the grid input description.
    """
    if isinstance(grid_inp, int):
        # grid_inp as an int is deprecated (TODO: put a warning here)
        #        0,  1,  2,  3,  4,  5
        nr   = [20, 40, 60, 75, 100, 125][grid_inp]
        prec = [13, 17, 21, 29, 41, 59][grid_inp]
        radgrid = RadialGrid(nr, "chebyshev", "logm3",
                             dtype=dtype, device=device)
        return LebedevGrid(radgrid, prec=prec)

    elif isinstance(grid_inp, str):
        grid_str = grid_inp.lower().replace("-", "")
        grid_cls = _get_grid_cls(grid_str)
        return grid_cls(atomz, dtype=dtype, device=device)
    else:
        raise TypeError("Unknown type of grid_inp: %s" % type(grid_inp))


def _get_grid_cls(grid_str: str):
    # returns the grid class given the string
    return GRID_CLS_MAP[grid_str]
