import torch
from abc import abstractmethod

class BaseContractedGaussian(object):
    def __init__(self, dtype, device):
        self._dtype = dtype
        self._device = device

    @property
    def dtype(self):
        return self._dtype

    @property
    def device(self):
        return self._device

    @abstractmethod
    def loadbasis(self, atomz, cartesian):
        """
        Given atomz, returns the parameters needed for the basis.
        If cartesian == True, returns (ijks, alphas, coeffs, nelmts).
        else, returns (lms, alphas, coeffs, nelmts).
        The elements in return should be torch.tensor
        """
        pass

    def construct_basis(self, atomzs, atomposs, cartesian):
        # atomzs: torch.tensor int (natoms)
        # atomposs: torch.tensor (natoms, 3)
        # returns: (ijks, alphas, coeffs, nelmts, poss) each are torch.tensor
        i = 0
        for atomz, atompos in zip(atomzs, atomposs):
            ijks, alphas, coeffs, nelmts = self.loadbasis(int(atomz), cartesian=cartesian)
            atpos = atompos.unsqueeze(0).repeat((nelmts.sum(), 1))
            if i == 0:
                all_ijks = ijks
                all_alphas = alphas
                all_coeffs = coeffs
                all_nelmts = nelmts
                all_poss = atpos
            else:
                all_ijks = torch.cat((all_ijks, ijks), dim=0)
                all_alphas = torch.cat((all_alphas, alphas), dim=0)
                all_coeffs = torch.cat((all_coeffs, coeffs), dim=0)
                all_nelmts = torch.cat((all_nelmts, nelmts), dim=0)
                all_poss = torch.cat((all_poss, atpos), dim=0)
            i = 1
        return all_ijks, all_alphas, all_coeffs, all_nelmts, all_poss

def normalize_basisname(basisname):
    b = basisname.lower()
    b = b.replace("+", "p")
    b = b.replace("*", "s")
    return b

def to_angmom(spdf):
    return {
        "s": 0,
        "p": 1,
        "d": 2,
        "f": 3
    }[spdf.lower()]

def expand_basis(spdfs, all_alphas, all_coeffs, cartesian):
    # spdfs: list (cgto) of the orbital name
    # all_alphas, all_coeffs: list (cgto) of list (elemental gaussian)
    ijks = []
    new_alphas = []
    new_coeffs = []
    for icgto in range(len(all_alphas)):
        spdf = spdfs[icgto].lower()
        alphas = all_alphas[icgto]
        coeffs = all_coeffs[icgto]
        nelmts = len(alphas)

        nrep, ijklist = get_angmom_spec(spdf, cartesian)
        new_alphas += [alphas] * nrep
        new_coeffs += [coeffs] * nrep
        for ijk in ijklist:
            ijks += [(ijk * (nelmts//len(ijk)))]

    return ijks, new_alphas, new_coeffs

def get_angmom_spec(spdf, cartesian):
    nrep = {
        "s": [1,1],
        "p": [3,3],
        "sp": [3,3],
        "d": [5,6],
        "f": [7,10],
        "g": [9,15],
        "h": [11,21],
    }[spdf][1 if cartesian else 0]
    if cartesian:
        ijk_list = {
            "s": [[(0,0,0)]],
            "p": [[(0,0,1)], [(0,1,0)], [(1,0,0)]],
            "d": [[(0,0,2)], [(0,2,0)], [(2,0,0)], [(1,1,0)], [(1,0,1)], [(0,1,1)]],
            "f": [[(0,0,3)], [(0,3,0)], [(3,0,0)], [(2,1,0)], [(1,2,0)], [(2,0,1)],
                  [(1,0,2)], [(0,2,1)], [(0,1,2)], [(1,1,1)]],
            "sp": [[(0,0,0), (0,0,1)], [(0,0,0), (0,1,0)], [(0,0,0), (1,0,0)]],
        }[spdf]
    else:
        ijk_list = {
            "s": [[(0,0)]],
            "p": [[(1,0)], [(1,-1)], [(1,1)]],
            "d": [[(2,0)], [(2,-1)], [(2,1)], [(2,-2)], [(2,2)]],
            "f": [[(3,0)], [(3,-1)], [(3,1)], [(3,-2)], [(3,2)], [(3,-3)], [(3,3)]],
            "sp": [[(0,0),(1,0)], [(0,0),(1,-1)], [(0,0),(1,1)]],
        }[spdf]
    return nrep, ijk_list

def flatten(array2d):
    res = []
    for elmt in array2d:
        res += elmt
    return res
