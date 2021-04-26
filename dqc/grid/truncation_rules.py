from abc import abstractmethod
from typing import List
import torch
from dqc.grid.radial_grid import RadialGrid

class BaseTruncationRules(object):
    """
    Base class to store the truncation rules of an individual atomic grid.
    """
    @abstractmethod
    def to_truncate(self, atz: int) -> bool:
        # decide whether to truncate the atom's grid
        pass

    @abstractmethod
    def rad_slices(self, atz: int, radgrid: RadialGrid) -> List[slice]:
        # get the list of slices of radial grid
        pass

    @abstractmethod
    def precs(self, atz: int, radgrid: RadialGrid) -> List[int]:
        # get the list of precisions of angular grid for each slice in the
        # sliced radial grids
        pass

class NoTrunc(BaseTruncationRules):
    def __init__(self):
        pass

    def to_truncate(self, atz: int) -> bool:
        return False

    def rad_slices(self, atz: int, radgrid: RadialGrid) -> List[slice]:
        raise RuntimeError("This shouldn't be called. Report to Github")

    def precs(self, atz: int, radgrid: RadialGrid) -> List[int]:
        raise RuntimeError("This shouldn't be called. Report to Github")

class DasguptaTrunc(BaseTruncationRules):
    """
    Truncation rule from Dasgupta et al., https://onlinelibrary.wiley.com/doi/epdf/10.1002/jcc.24761
    """
    def __init__(self, nr: int):
        if nr == 75:
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
        elif nr == 99:
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
        else:
            msg = ("Dasgupta truncation can only accept radial grid with number of points in %s" %
                   (str([75, 99])))
            raise ValueError(msg)
        self._truncate_idxs = truncate_idxs
        self._truncate_precs = truncate_precs

    def to_truncate(self, atz: int) -> bool:
        # decide whether to truncate the atom's grid
        return atz in self._truncate_idxs

    def rad_slices(self, atz: int, radgrid: RadialGrid) -> List[slice]:
        # get the list of slices of radial grid
        idxs = self._truncate_idxs[atz]
        return [slice(idxs[i], idxs[i + 1], None) for i in range(len(idxs) - 1)]

    def precs(self, atz: int, radgrid: RadialGrid) -> List[int]:
        # get the list of precisions of angular grid for each slice in the
        # sliced radial grids
        return self._truncate_precs[atz]

class NWChemTrunc(BaseTruncationRules):
    """
    NWChem truncation rules.
    From https://github.com/pyscf/pyscf/blob/18030c75a5c69c1da84574d111693074a622de56/pyscf/dft/gen_grid.py#L122
    """
    def __init__(self, radii_list: List[float],
                 prec: int,
                 precs_list: List[int],
                 dtype: torch.dtype,
                 device: torch.device):
        self._radii_list = radii_list
        self._alphas = torch.tensor([
            [0.25, 0.5, 1.0, 4.5],
            [0.1667, 0.5, 0.9, 3.5],
            [0.1, 0.4, 0.8, 2.5],
        ], dtype=dtype, device=device)
        self._prec = prec
        precs_list = precs_list[4:]

        if prec == 13:
            precs_idxs = [1, 2, 2, 2, 1]
            self._precs = [precs_list[ii] for ii in precs_idxs]
        elif prec >= 13:
            idx: int = precs_list.index(prec)
            precs_idxs = [1, 3, idx - 1, idx, idx - 1]
            self._precs = [precs_list[ii] for ii in precs_idxs]

    def to_truncate(self, atz: int) -> bool:
        if self._prec < 13:
            return False
        return True

    def rad_slices(self, atz: int, radgrid: RadialGrid) -> List[slice]:
        ratom = self._radii_list[atz]
        ralphas = self._alphas * ratom
        rgrid = radgrid.get_rgrid().reshape(-1, 1)  # (nr, 1)
        if atz <= 2:  # H & He
            ralphas_i = ralphas[0]
        elif atz <= 10:
            ralphas_i = ralphas[1]
        else:
            ralphas_i = ralphas[2]
        # place has value from 0 to 4 (inclusive)
        place = torch.sum(rgrid > ralphas_i, dim=-1)  # (nr,)

        # convert it to slice
        pl, counts = torch.unique_consecutive(place, return_counts=True)
        idx = 0
        res: List[slice] = []
        for i in range(len(self._precs)):
            c = int(counts[i])
            res.append(slice(idx, idx + c, None))
            idx += c
        return res

    def precs(self, atz: int, radgrid: RadialGrid) -> List[int]:
        # get the list of precisions of angular grid for each slice in the
        # sliced radial grids
        return self._precs
