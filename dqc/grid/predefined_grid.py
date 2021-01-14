import torch
from typing import List
from dqc.grid.base_grid import BaseGrid
from dqc.grid.radial_grid import RadialGrid, DE2Transformation
from dqc.grid.lebedev_grid import TruncatedLebedevGrid

# this file contains predefined atomic grids from literatures

__all__ = ["SG2", "SG3"]

_dtype = torch.double
_device = torch.device("cpu")

_ATOM_RADIUS = {
    1: 1.0,
    2: 0.927272,
    3: 3.873661,
    4: 2.849396,
    5: 2.204757,
    6: 1.714495,
    7: 1.409631,
    8: 1.232198,
    9: 1.084786,
    10: 0.965273,
    11: 4.208762,
    12: 3.252938,
    13: 3.433889,
    14: 2.752216,
    15: 2.322712,
    16: 2.060717,
    17: 1.842024,
    18: 1.662954,
    19: 5.243652,
    20: 4.218469,
    21: 3.959716,
    22: 3.778855,
    23: 3.626288,
    24: 3.675012,
    25: 3.381917,
    26: 3.258487,
    27: 3.153572,
    28: 3.059109,
    29: 3.330979,
    30: 2.897648,
    31: 3.424103,
    32: 2.866859,
    33: 2.512233,
    34: 2.299617,
    35: 2.111601,
    36: 1.951590,
    37: 5.631401,
    38: 4.632850,
    39: 4.299870,
    40: 4.091705,
    41: 3.985219,
    42: 3.841740,
    43: 3.684647,
    44: 3.735235,
    45: 3.702057,
    46: 1.533028,
    47: 3.655961,
    48: 3.237216,
    49: 3.777242,
    50: 3.248093,
    51: 2.901067,
    52: 2.691328,
    53: 2.501704,
    54: 2.337950,
}
def get_atomic_radius(atomz: int) -> float:
    # returns the atomic radius for the given atomz
    return _ATOM_RADIUS[atomz]

class SG2(TruncatedLebedevGrid):
    """
    SG2 grid from https://onlinelibrary.wiley.com/doi/epdf/10.1002/jcc.24761
    """
    nr = 75
    prec = 29
    # ref: Table 1 from https://onlinelibrary.wiley.com/doi/epdf/10.1002/jcc.24761
    de2_alphas = {
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
    }
    truncate_idxs = {
        1: [0, 35, 47, 63, 70, 75],
        3: [0, 35, 47, 64, 71, 75],
        4: [0, 35, 47, 64, 71, 75],
        5: [0, 35, 47, 64, 71, 75],
        6: [0, 35, 47, 64, 71, 75],
        7: [0, 35, 47, 64, 71, 75],
        8: [0, 30, 44, 62, 70, 75],
        9: [0, 26, 42, 61, 69, 75],
        11: [0, 35, 47, 64, 71, 75],
        12: [0, 35, 47, 64, 71, 75],
        13: [0, 32, 47, 64, 71, 75],
        14: [0, 32, 47, 64, 71, 75],
        15: [0, 30, 44, 61, 68, 75],
        16: [0, 30, 44, 61, 68, 75],
        17: [0, 26, 42, 61, 69, 75],
    }
    truncate_precs = {
        1: [3, 17, 29, 15, 7],
        3: [3, 17, 29, 15, 11],
        4: [3, 17, 29, 15, 11],
        5: [3, 17, 29, 19, 7],
        6: [3, 17, 29, 19, 7],
        7: [3, 17, 29, 15, 7],
        8: [3, 17, 29, 19, 11],
        9: [3, 17, 29, 17, 11],
        11: [3, 17, 29, 15, 11],
        12: [3, 17, 29, 15, 11],
        13: [3, 17, 29, 19, 11],
        14: [3, 17, 29, 19, 11],
        15: [3, 17, 29, 19, 9],
        16: [3, 17, 29, 19, 9],
        17: [3, 17, 29, 17, 11],
    }

    def __init__(self, atomz: int, dtype: torch.dtype = _dtype,
                 device: torch.device = _device):
        # prepare the whole radial grid
        ratom = get_atomic_radius(atomz)
        grid_transform = DE2Transformation(
            alpha=self.de2_alphas.get(atomz, 1.0),
            rmin=1e-7, rmax=15 * ratom)
        radgrid = RadialGrid(
            self.nr, grid_integrator="uniform", grid_transform=grid_transform,
            dtype=dtype, device=device)

        is_truncated = self._is_truncated(atomz)
        if is_truncated:
            # truncated
            slices = self._get_truncate_slices(atomz)
            precs = self._get_truncate_precs(atomz)
            assert len(slices) == len(precs), "Please report this bug to the github page"
            # list of radial grid slices
            radgrids: List[BaseGrid] = [radgrid[sl] for sl in slices]
        else:
            # not truncated
            radgrids = [radgrid]
            precs = [self.prec]
        super().__init__(radgrids, precs)

    def _is_truncated(self, atomz: int) -> bool:
        # check if atomz has a truncated grid
        return atomz in self.truncate_idxs

    def _get_truncate_slices(self, atomz: int) -> List[slice]:
        # list of slices in radial grid in the truncated grid
        idxs = self.truncate_idxs[atomz]
        return [slice(idxs[i], idxs[i + 1], None) for i in range(len(idxs) - 1)]

    def _get_truncate_precs(self, atomz: int) -> List[int]:
        # list of angular precisions in the truncated grid
        return self.truncate_precs[atomz]

class SG3(SG2):
    """
    SG3 grid from https://onlinelibrary.wiley.com/doi/epdf/10.1002/jcc.24761
    """
    nr = 99
    prec = 41
    # ref: Table 1 from https://onlinelibrary.wiley.com/doi/epdf/10.1002/jcc.24761
    de2_alphas = {
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
    }
    truncate_idxs = {
        1: [0, 45, 61, 82, 92, 99],
        3: [0, 46, 62, 84, 93, 99],
        4: [0, 42, 48, 62, 84, 87, 93, 99],
        5: [0, 42, 48, 62, 84, 93, 99],
        6: [0, 46, 62, 84, 85, 87, 93, 99],
        7: [0, 40, 58, 82, 93, 99],
        8: [0, 40, 54, 56, 58, 82, 83, 84, 92, 99],
        9: [0, 35, 52, 56, 81, 83, 91, 99],
        11: [0, 46, 62, 84, 93, 99],
        12: [0, 48, 63, 83, 90, 99],
        13: [0, 42, 48, 62, 84, 87, 93, 99],
        14: [0, 42, 48, 62, 84, 93, 99],
        15: [0, 35, 36, 54, 58, 83, 85, 93, 99],
        16: [0, 35, 36, 54, 58, 83, 85, 93, 99],
        17: [0, 35, 52, 56, 81, 83, 91, 99],
    }
    truncate_precs = {
        1: [3, 17, 41, 23, 11],
        3: [3, 17, 41, 19, 11],
        4: [3, 15, 17, 41, 23, 19, 11],
        5: [3, 15, 17, 41, 23, 11],
        6: [3, 19, 41, 29, 23, 19, 15],
        7: [3, 17, 41, 19, 11],
        8: [3, 17, 23, 29, 41, 29, 23, 19, 11],
        9: [3, 17, 23, 41, 23, 17, 11],
        11: [3, 17, 41, 19, 11],
        12: [3, 17, 41, 19, 11],
        13: [3, 15, 17, 41, 23, 19, 11],
        14: [3, 15, 17, 41, 23, 11],
        15: [3, 15, 17, 23, 41, 23, 19, 11],
        16: [3, 15, 17, 23, 41, 23, 19, 11],
        17: [3, 17, 23, 41, 23, 17, 11],
    }
