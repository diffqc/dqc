from abc import abstractmethod

class BaseContractedGaussian(object):
    @abstractmethod
    def loadbasis(self, atomz, cartesian):
        """
        Given atomz, returns the parameters needed for the basis.
        If cartesian == True, returns (ijks, alphas, coeffs, nelmts).
        """
        pass

    def get_spherical(self, atomzs, atompos):
        pass

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
