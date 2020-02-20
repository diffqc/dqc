import torch
import numpy as np
from ddft.dft.dft import DFT, DFTMulti
from ddft.hamiltons.hatomradial import HamiltonAtomRadial
from ddft.grids.radialshiftexp import RadialShiftExp
from ddft.modules.equilibrium import EquilibriumModule
from ddft.eks import BaseEKS, Hartree, xLDA

__all__ = ["atom"]

def atom(atomz, eks_model="lda",
         gwmin=1e-5, gwmax=1e2, ng=100,
         rmin=1e-6, rmax=1e4, nr=2000,
         dtype=torch.float64, device="cpu"):

    eig_options = {
        "method": "exacteig",
    }

    # normalize the device and eks_model
    device = _normalize_device(device)
    eks_model = _normalize_eks(eks_model)

    # get the atomic configuration
    orbitals = Orbitals(atomz, dtype, device)

    is_radial = _check_atom_is_radial(atomz)
    if not is_radial:
        raise RuntimeError("Non-radial atom is not supported yet.")
    else:
        # setup the grids and the hamiltonians
        gwidths = torch.logspace(np.log10(gwmin), np.log10(gwmax), ng, dtype=dtype).to(device)
        grid = RadialShiftExp(rmin, rmax, nr, dtype=dtype, device=device)
        angmoms = orbitals.get_angmoms()
        H_models = [HamiltonAtomRadial(grid, gwidths, angmom=angmom).to(dtype).to(device) for angmom in angmoms]

        # setup the hamiltonian parameters and the occupation numbers
        atomz_tensor = torch.tensor([atomz]).to(dtype).to(device)
        hparams = [atomz_tensor]
        vext = torch.zeros_like(grid.rgrid[:,0]).unsqueeze(0).to(dtype).to(device)

        # setup the modules
        nlowests = orbitals.get_nlowests()
        all_eks_models = Hartree(grid)
        if eks_model is not None:
            all_eks_models = all_eks_models + eks_model
        dft_model = DFTMulti(H_models, all_eks_models, nlowests, **eig_options)
        scf_model = EquilibriumModule(dft_model)

        # calculate the density
        foccs = orbitals.get_foccs()
        density0 = torch.zeros_like(vext).to(device)
        all_hparams = [hparams for _ in range(len(H_models))]
        density = scf_model(density0, vext, foccs, all_hparams)
        energy = dft_model.energy()

        return energy, density

class Orbitals(object):
    def __init__(self, atomz, dtype, device):
        self.atomz = atomz
        self.elocc, self.elangmom = get_occupation(atomz)

        # get angular momentums
        self.max_angmom = np.max(self.elangmom)
        self.angmoms = np.arange(self.max_angmom+1)

        # calculate the occupation numbers
        foccs = []
        nlowests = []
        occ_nums = [2, 6, 10, 14]
        for angmom in self.angmoms:
            eltot = np.sum(self.elocc[self.elangmom == angmom])
            occ = occ_nums[angmom]
            nlowest = int(np.ceil(eltot*1.0/occ))
            focc = torch.ones(nlowest, device=device, dtype=dtype).unsqueeze(0) * occ
            if eltot % occ != 0:
                focc[-1] = 1.0 * (eltot % occ)
            foccs.append(focc)
            nlowests.append(nlowest)

        self.foccs = foccs
        self.nlowests = nlowests

    def get_angmoms(self):
        return self.angmoms

    def get_focc(self):
        return self.focc

    def get_nlowests(self):
        return self.nlowests

    def get_foccs(self):
        return self.foccs

def get_occupation(atomz):
    orbitals = ["1s", "2s", "2p", "3s", "3p", "4s", "3d", "4p", "5s", "4d", "5p", "6s", "4f"]
    idx = 0
    elocc = []
    elangmom = []
    while atomz > 0:
        orbital = orbitals[idx]
        occ, angmom = get_max_occ_angmom(orbital)
        if atomz < occ:
            occ = atomz
        elocc.append(occ)
        elangmom.append(angmom)
        atomz = atomz - occ
        idx = idx + 1
    return np.array(elocc), np.array(elangmom)

def get_max_occ_angmom(orbital):
    return {
        "s": (2, 0),
        "p": (6, 1),
        "d": (10, 2),
        "f": (14, 3)
    }[orbital[-1]]

def _check_atom_is_radial(atomz):
    return atomz in [1,2,3,4,10,18,36,54,86]

def _normalize_device(device):
    if isinstance(device, str):
        return torch.device(device)
    elif isinstance(device, torch.device):
        return device
    else:
        raise TypeError("Unknown type of device: %s" % type(device))

def _normalize_eks(eks):
    if isinstance(eks, str):
        ek = eks.lower()
        if ek == "lda":
            return xLDA()
        else:
            raise RuntimeError("Unknown EKS model: %s" % eks)
    elif isinstance(eks, BaseEKS):
        return eks
    else:
        raise RuntimeError("Unknown EKS input type: %s" % type(eks))

if __name__ == "__main__":
    energy, density = atom(3)
    print(energy)
