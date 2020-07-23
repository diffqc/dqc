import os
import torch
from ddft.basissets.base_basisset import BaseAtomicBasis
from ddft.hamiltons.hmolcgauss import HamiltonMoleculeCGauss

class CartCGTOBasis(BaseAtomicBasis):
    def __init__(self, atomz, basisname, requires_grad=True,
                 dtype=torch.float32, device=torch.device('cpu')):
        self._dtype = dtype
        self._device = device
        self.atomz = atomz
        self.requires_grad = requires_grad

        # file name format of the database
        basisname = normalize_basisname(basisname)
        thisdir = os.path.dirname(os.path.realpath(__file__))
        fnameformat = os.path.join(thisdir, "database", basisname, "%02d.gaussian94")
        self.fnameformat = fnameformat

        # ijk: int (nbasis,)
        # alphas: (nbasis,)
        # coeffs: (nbasis,)
        # nelmts: int (ncgto,) containing the number of gaussians in the contracted basis
        fname = self.fnameformat % atomz
        self.ijk, self.alphas, self.coeffs, self.nelmts = self._loadbasis(fname)

        if requires_grad:
            self.alphas = self.alphas.requires_grad_()
            self.coeffs = self.coeffs.requires_grad_()

    def _loadbasis(self, file):
        if not os.path.exists(file):
            raise RuntimeError("The file for basis is not found: %s" % (file))

        # read the content
        with open(file, "r") as f:
            lines = f.read().split("\n")

        # skip the header
        while True:
            line = lines.pop(0)
            if line == "": continue
            if line.startswith("!"): continue
            break

        spdfs = []
        alphas = []
        angmoms = []
        coeffs = []
        while len(lines) > 0:
            line = lines.pop(0)
            if line.startswith("**"): break
            spdf, nelmt_str, _ = line.split()
            spdf = spdf.lower()
            nelmt = int(nelmt_str)

            all_elmts = []
            for i in range(nelmt):
                line = lines.pop(0)
                line = line.replace("D", "E")
                elmts = [float(s) for s in line.split()]
                all_elmts.append(elmts)

            for i,spdf_letter in enumerate(spdf):
                alpha = []
                coeff = []
                angmom = []
                for j in range(nelmt):
                    elmts = all_elmts[j]
                    alpha.append(elmts[0])
                    coeff.append(elmts[1+i])
                    angmom.append(to_angmom(spdf_letter))
                alphas.append(alpha)
                coeffs.append(coeff)
                angmoms.append(angmom)
                spdfs.append(spdf_letter)

        # save the results
        ijks, alphas, coeffs = expand_basis(spdfs, alphas, coeffs)
        nelmts = torch.tensor([len(ijk) for ijk in ijks], dtype=torch.int32, device=self.device)
        ijks   = torch.tensor(flatten(ijks), dtype=torch.int32, device=self.device)
        alphas = torch.tensor(flatten(alphas), dtype=self.dtype, device=self.device)
        coeffs = torch.tensor(flatten(coeffs), dtype=self.dtype, device=self.device)
        res = (ijks, alphas, coeffs, nelmts)
        return res

    def _get_basis_params(self):
        return (self.ijk, self.alphas, self.coeffs, self.nelmts)

    @property
    def dtype(self):
        return self._dtype

    @property
    def device(self):
        return self._device

    @staticmethod
    def construct_hamiltonian(grid, bases_list, atomposs):
        # bases_list is a list of objects of this class with length (natoms,)
        # atomposs: torch.tensor (natoms, ndim)
        i = 0
        for basis,atompos in zip(bases_list,atomposs):
            ijks, alphas, coeffs, nelmts = basis._get_basis_params()
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

        dtype = bases_list[0].dtype
        device = bases_list[0].device
        atomzs = torch.tensor([basis.atomz for basis in bases_list], dtype=dtype, device=device)
        H_model = HamiltonMoleculeCGauss(grid, all_ijks, all_alphas,
            all_poss, all_coeffs, all_nelmts, atomposs,
            atomzs).to(dtype).to(device)
        return H_model

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

def expand_basis(spdfs, all_alphas, all_coeffs):
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

        nrep, ijklist = get_angmom_spec(spdf)
        new_alphas += [alphas] * nrep
        new_coeffs += [coeffs] * nrep
        for ijk in ijklist:
            ijks += [(ijk * (nelmts//len(ijk)))]

    return ijks, new_alphas, new_coeffs

def get_angmom_spec(spdf):
    nrep = {
        "s": 1,
        "p": 3,
        "sp": 3,
        "d": 6,
        "f": 10,
        "g": 15,
        "h": 21,
    }[spdf]
    ijk_list = {
        "s": [[(0,0,0)]],
        "p": [[(0,0,1)], [(0,1,0)], [(1,0,0)]],
        "d": [[(0,0,2)], [(0,2,0)], [(2,0,0)], [(1,1,0)], [(1,0,1)], [(0,1,1)]],
        "f": [[(0,0,3)], [(0,3,0)], [(3,0,0)], [(2,1,0)], [(1,2,0)], [(2,0,1)],
              [(1,0,2)], [(0,2,1)], [(0,1,2)], [(1,1,1)]],
        "g": [[(0,0,4)], [(0,4,0)], [(4,0,0)], [(3,1,0)], [(3,0,1)], [(2,2,0)],
              [(2,0,2)], [(2,1,1)], [(1,3,0)], [(1,0,3)], [(1,2,1)], [(1,1,2)],
              [(0,3,1)], [(0,1,3)], [(0,2,2)]],
        "sp": [[(0,0,0), (0,0,1)], [(0,0,0), (0,1,0)], [(0,0,0), (1,0,0)]],
    }[spdf]
    return nrep, ijk_list

def flatten(array2d):
    res = []
    for elmt in array2d:
        res += elmt
    return res

if __name__ == "__main__":
    dtype = torch.float64
    basis = CartCGTOBasis(1, "6-311++G**", dtype=dtype)
    ijks, alphas, coeffs, nelmts = basis._get_basis_params()
    print("IJKs:")
    print(ijks)
    print("Alphas:")
    print(alphas)
    print("Coeffs:")
    print(coeffs)
    print("Nelmts:")
    print(nelmts)
