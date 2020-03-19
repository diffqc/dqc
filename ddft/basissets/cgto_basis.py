import os
from ddft.basissets.base_basisset import BaseContractedGaussian, \
    normalize_basisname, to_angmom, expand_basis, flatten

class CGTOBasis(BaseContractedGaussian):
    def __init__(self, basisname, fnameformat=None):
        """
        Read the database of basis from files with name format given in `fnameformat`
        """
        super(CGTOBasis, self).__init__()

        basisname = normalize_basisname(basisname)
        if fnameformat is None:
            thisdir = os.path.dirname(os.path.realpath(__file__))
            fnameformat = os.path.join(thisdir, "database", basisname, "%02d.gaussian94")
        self.fnameformat = fnameformat
        self.res_memory = {}

    def loadbasis(self, atomz, cartesian):
        if atomz in self.res_memory:
            return self.res_memory[atomz]

        file = self.fnameformat % atomz
        if not os.path.exists(file):
            raise RuntimeError("The file for Z=%d is not found: %s" % (atomz, file))

        # read the content
        with open(file, "r") as f:
            # the content starts from line 15
            lines = f.read().split("\n")[14:]

        spdfs = []
        alphas = []
        angmoms = []
        coeffs = []
        while len(lines) > 0:
            line = lines.pop(0)
            if line.startswith("**"): break
            spdf, nelmt_str, _ = line.split()
            spdfs.append(spdf)
            nelmt = int(nelmt_str)

            alpha = []
            coeff = []
            angmom = []
            for i in range(nelmt):
                line = lines.pop(0)
                elmts = [float(s) for s in line.split()]
                if len(elmts) == 2:
                    alpha.append(elmts[0])
                    coeff.append(elmts[1])
                    angmom.append(to_angmom(spdf))
                elif len(elmts) == 3:
                    alpha.append(elmts[0])
                    coeff.append(elmts[1])
                    angmom.append(to_angmom(spdf[0]))
                    alpha.append(elmts[0])
                    coeff.append(elmts[2])
                    angmom.append(to_angmom(spdf[1]))
            alphas.append(alpha)
            coeffs.append(coeff)
            angmoms.append(angmom)

        # save the results
        ijks, alphas, coeffs = expand_basis(spdfs, alphas, coeffs, cartesian=cartesian)
        nelmts = [len(ijk) for ijk in ijks]
        ijks = flatten(ijks)
        alphas = flatten(alphas)
        coeffs = flatten(coeffs)
        res = (ijks, alphas, coeffs, nelmts)
        self.res_memory[atomz] = res
        return res

if __name__ == "__main__":
    basis = CGTOBasis("6-31++G**")
    ijks, alphas, coeffs, nelmts = basis.loadbasis(3, cartesian=True)
    # ijks, alphas, pos, coeffs, nelmts = basis.get_cartesian(atomzs)#, atompos)
    print("IJKs:")
    print(ijks)
    print("Alphas:")
    print(alphas)
    print("Coeffs:")
    print(coeffs)
    print("Nelmts:")
    print(nelmts)
