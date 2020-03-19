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

        if spdf == "s":
            ijks += [ ([(0,0,0)]*nelmts) ]
            new_alphas += [alphas]
            new_coeffs += [coeffs]
        elif spdf == "p":
            ijks += [([(0,0,1)]*nelmts)] + \
                    [([(0,1,0)]*nelmts)] + \
                    [([(1,0,0)]*nelmts)]
            new_alphas += [alphas] * 3
            new_coeffs += [coeffs] * 3
        elif spdf == "d":
            ijks += [([(0,0,2)]*nelmts)] + \
                    [([(0,2,0)]*nelmts)] + \
                    [([(2,0,0)]*nelmts)] + \
                    [([(1,1,0)]*nelmts)] + \
                    [([(1,0,1)]*nelmts)] + \
                    [([(0,1,1)]*nelmts)]
            new_alphas += [alphas]*6
            new_coeffs += [coeffs]*6
        elif spdf == "f":
            ijks += [([(0,0,3)]*nelmts)] + \
                    [([(0,3,0)]*nelmts)] + \
                    [([(3,0,0)]*nelmts)] + \
                    [([(2,1,0)]*nelmts)] + \
                    [([(1,2,0)]*nelmts)] + \
                    [([(2,0,1)]*nelmts)] + \
                    [([(1,0,2)]*nelmts)] + \
                    [([(0,2,1)]*nelmts)] + \
                    [([(0,1,2)]*nelmts)] + \
                    [([(1,1,1)]*nelmts)]
            new_alphas += [alphas]*10
            new_coeffs += [coeffs]*10
        elif spdf == "sp":
            ijks += [([(0,0,0), (0,0,1)]*(nelmts//2))] + \
                    [([(0,0,0), (0,1,0)]*(nelmts//2))] + \
                    [([(0,0,0), (1,0,0)]*(nelmts//2))]
            new_alphas += [alphas]*3
            new_coeffs += [coeffs]*3
    return ijks, new_alphas, new_coeffs

def flatten(array2d):
    res = []
    for elmt in array2d:
        res += elmt
    return res
