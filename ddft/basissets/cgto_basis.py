import os
import torch
from ddft.basissets.base_basisset import BaseBasisModule
from ddft.hamiltons.hmolcgauss import HamiltonMoleculeCGauss

class CGTOBasis(BaseBasisModule):
    def __init__(self, basisname, cartesian=True, requires_grad=False,
                 fnameformat=None, dtype=torch.float32, device=torch.device('cpu')):
        """
        Read the database of basis from files with name format given in `fnameformat`
        """
        super(CGTOBasis, self).__init__()
        self._dtype = dtype
        self._device = device
        self.cartesian = cartesian

        basisname = normalize_basisname(basisname)
        if fnameformat is None:
            thisdir = os.path.dirname(os.path.realpath(__file__))
            fnameformat = os.path.join(thisdir, "database", basisname, "%02d.gaussian94")
        self.fnameformat = fnameformat
        self.res_memory = {}

        # cache
        self._basis_constructed = False

    @property
    def dtype(self):
        return self._dtype

    @property
    def device(self):
        return self._device

    def loadbasis(self, atomz, cartesian):
        if atomz in self.res_memory:
            return self.res_memory[atomz]

        file = self.fnameformat % atomz
        if not os.path.exists(file):
            raise RuntimeError("The file for Z=%d is not found: %s" % (atomz, file))

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
        ijks, alphas, coeffs = expand_basis(spdfs, alphas, coeffs, cartesian=cartesian)
        nelmts = torch.tensor([len(ijk) for ijk in ijks], dtype=torch.int32, device=self.device)
        ijks   = torch.tensor(flatten(ijks), dtype=torch.int32, device=self.device)
        alphas = torch.tensor(flatten(alphas), dtype=self.dtype, device=self.device)
        coeffs = torch.tensor(flatten(coeffs), dtype=self.dtype, device=self.device)
        res = (ijks, alphas, coeffs, nelmts)
        self.res_memory[atomz] = res
        return res

    def construct_basis(self, atomzs, atomposs, requires_grad=False):
        # atomzs: torch.tensor int (natoms)
        # atomposs: torch.tensor (natoms, 3)
        # returns: (ijks, alphas, coeffs, nelmts, poss) each are torch.tensor
        i = 0
        self.atomzs = atomzs
        self.atomposs = atomposs
        cartesian = self.cartesian
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

        if requires_grad:
            self.all_alphas = torch.nn.Parameter(all_alphas)
            self.all_coeffs = torch.nn.Parameter(all_coeffs)
            self.all_poss = torch.nn.Parameter(all_poss)
        else:
            self.all_alphas = all_alphas
            self.all_coeffs = all_coeffs
            self.all_poss = all_poss

        self.all_nelmts = all_nelmts
        self.all_ijks = all_ijks
        self._basis_constructed = True

    def is_basis_constructed(self):
        return self._basis_constructed

    def get_hamiltonian(self, grid):
        if not self.is_basis_constructed():
            raise RuntimeError("Must call construct_basis before calling get_hamiltonian")
        H_model = HamiltonMoleculeCGauss(grid, self.all_ijks, self.all_alphas,
            self.all_poss, self.all_coeffs, self.all_nelmts, self.atomposs,
            self.atomzs).to(self.dtype).to(self.device)
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

if __name__ == "__main__":
    dtype = torch.float64
    basis = CGTOBasis("6-31++G**", dtype=dtype)
    atomzs = torch.tensor([1.0, 1.0], dtype=dtype)
    atomposs = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    # ijks, alphas, coeffs, nelmts = basis.loadbasis(3, cartesian=True)
    ijks, alphas, coeffs, nelmts, pos = basis.construct_basis(atomzs, atomposs, cartesian=True)
    print("IJKs:")
    print(ijks)
    print("Alphas:")
    print(alphas)
    print("Coeffs:")
    print(coeffs)
    print("Nelmts:")
    print(nelmts)
    print("Positions:")
    print(pos)
