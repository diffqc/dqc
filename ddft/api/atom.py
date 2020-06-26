import torch
import numpy as np
from ddft.dft.dft import DFT, DFTMulti
from ddft.utils.misc import set_default_option
from ddft.hamiltons.hatomradial import HamiltonAtomRadial
from ddft.hamiltons.hatomygauss import HamiltonAtomYGauss
from ddft.grids.radialgrid import LegendreRadialShiftExp
from ddft.grids.sphangulargrid import Lebedev
from ddft.modules.equilibrium import EquilibriumModule
from ddft.eks import BaseEKS, Hartree, xLDA

__all__ = ["atom"]

def atom(atomz, charge=0,
         is_radial=None,
         eks_model="lda",
         gwmin=1e-5, gwmax=1e3, ng=60,
         rmin=1e-5, rmax=1e2, nr=200,
         dtype=torch.float64, device="cpu",
         eig_options=None, scf_options=None, bck_options=None):

    eig_options = set_default_option({
        "method": "exacteig",
    }, eig_options)
    scf_options = set_default_option({
        "min_eps": 1e-9,
        "jinv0": 0.5,
        "alpha0": 1.0,
        "verbose": False,
        "method": "selfconsistent",
    }, scf_options)
    bck_options = set_default_option({
        "min_eps": 1e-9,
    }, bck_options)
    cylsymm = False
    coulexp = True

    # normalize the device and eks_model
    device = _normalize_device(device)
    eks_model = _normalize_eks(eks_model)

    # get the atomic configuration
    numel = atomz - charge
    if is_radial is None:
        is_radial = _check_atom_is_radial(numel)
    orbitals = Orbitals(numel, dtype, device,
        radial_symmetric=is_radial, cylindrical_symmetric=cylsymm)

    # set up the basis and the radial grid
    gwidths = torch.logspace(np.log10(gwmin), np.log10(gwmax), ng, dtype=dtype).to(device)
    radgrid = LegendreRadialShiftExp(rmin, rmax, nr, dtype=dtype, device=device)
    angmoms = orbitals.get_angmoms()

    # setup the grids and the hamiltonians
    if not is_radial:
        maxangmom = angmoms.max()
        grid = Lebedev(radgrid, prec=13, basis_maxangmom=maxangmom, dtype=dtype, device=device)
        H_models = [HamiltonAtomYGauss(grid, gwidths, maxangmom=maxangmom, coulexp=coulexp).to(dtype).to(device)]
    else:
        grid = radgrid
        H_models = [HamiltonAtomRadial(grid, gwidths, angmom=angmom, coulexp=coulexp).to(dtype).to(device) for angmom in angmoms]

    # setup the hamiltonian parameters and the occupation numbers
    atomz_tensor = torch.tensor([atomz]).to(dtype).to(device)
    charge_tensor = torch.tensor([charge]).to(dtype).to(device)
    hparams = [atomz_tensor, charge_tensor]
    vext = torch.zeros_like(grid.rgrid[:,0]).unsqueeze(0).to(dtype).to(device)

    # setup the modules
    nlowests = orbitals.get_nlowests()
    all_eks_models = Hartree()
    if eks_model is not None:
        all_eks_models = all_eks_models + eks_model
    all_eks_models.set_grid(grid)
    dft_model = DFTMulti(H_models, all_eks_models, nlowests, **eig_options)
    scf_model = EquilibriumModule(dft_model, forward_options=scf_options, backward_options=bck_options)

    # set up the dft object
    foccs = orbitals.get_foccs()
    density0 = torch.zeros_like(vext).to(device)
    all_hparams = [hparams for _ in range(len(H_models))]
    dft_model.set_vext(vext)
    dft_model.set_focc(foccs)
    dft_model.set_hparams(all_hparams)

    # calculate the density
    density0 = dft_model(density0).detach()
    density = scf_model(density0)
    energy = dft_model.energy(density)

    return energy, density

class Orbitals(object):
    def __init__(self, numel, dtype, device, radial_symmetric=True,
                 cylindrical_symmetric=True):
        self.numel = numel
        self.elocc, self.elangmom = get_occupation(numel)

        # get angular momentums
        self.max_angmom = np.max(self.elangmom)
        self.angmoms = np.arange(self.max_angmom+1)

        # calculate the occupation numbers
        if not radial_symmetric:
            if not cylindrical_symmetric:
                focc = []
                for elocc, elangmom in zip(self.elocc, self.elangmom):
                    maxocc = (2*elangmom + 1) * 2
                    if elocc == maxocc:
                        focc = focc + [2.0] * (2*elangmom+1)
                    else:
                        norb = (2*elangmom+1)
                        for m in range(norb):
                            nel = (elocc+norb - (m+1)) // norb
                            if nel == 0: break
                            focc.append(nel)
                foccs = [torch.tensor(focc, dtype=dtype, device=device).unsqueeze(0)]
                nlowests = [len(focc)]
            else:
                foccs = [torch.tensor(self.elocc, dtype=dtype, device=device).unsqueeze(0)]
                nlowests = [len(self.elocc)]
        else:
            foccs = []
            nlowests = []
            occ_nums = [2, 6, 10, 14]
            for angmom in self.angmoms:
                eltot = np.sum(self.elocc[self.elangmom == angmom])
                occ = occ_nums[angmom]
                nlowest = int(np.ceil(eltot*1.0/occ))
                focc = torch.ones(nlowest, device=device, dtype=dtype) * occ
                if eltot % occ != 0:
                    focc[-1] = 1.0 * (eltot % occ)
                foccs.append(focc.unsqueeze(0))
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

def get_occupation(numel):
    orbitals = ["1s", "2s", "2p", "3s", "3p", "4s", "3d", "4p", "5s", "4d", "5p", "6s", "4f"]
    idx = 0
    elocc = []
    elangmom = []
    while numel > 0:
        orbital = orbitals[idx]
        occ, angmom = get_max_occ_angmom(orbital)
        if numel < occ:
            occ = numel
        elocc.append(occ)
        elangmom.append(angmom)
        numel = numel - occ
        idx = idx + 1
    return np.array(elocc), np.array(elangmom)

def get_max_occ_angmom(orbital):
    return {
        "s": (2, 0),
        "p": (6, 1),
        "d": (10, 2),
        "f": (14, 3)
    }[orbital[-1]]

def _check_atom_is_radial(numel):
    return numel in [1,2,3,4,7,10,11,12,15,18,19,20,25,30,33,36,37,38,43,48,51,54,55,56,86] # 56 is 6s2

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
    import time
    from ddft.utils.safeops import safepow
    from ddft.utils.fd import finite_differences

    dtype = torch.float64
    class PseudoLDA(BaseEKS):
        def __init__(self, a, p):
            super(PseudoLDA, self).__init__()
            self.a = torch.nn.Parameter(a)
            self.p = torch.nn.Parameter(p)

        def forward(self, density):
            return self.a * safepow(density.abs(), self.p)

        def getfwdparams(self):
            return [self.a, self.p]

        def setfwdparams(self, *params):
            self.a = params[0]
            self.p = params[1]
            return 2

    # experimental data
    natoms = 1
    atomzs = [16,2,4,10,12,18,20,30,36][:natoms]
    expdata = torch.tensor([-2.904601242647059, -14.674582216911766,
                            -129.10649963235295, -200.32232950000002,
                            -529.4431757977941, -680.2348059195,
                            -1794.8478644, -2789.1691707830882][:natoms]).to(dtype)

    # pseudo-lda eks model
    a = torch.tensor([-0.7385587663820223]).to(dtype).requires_grad_()
    p = torch.tensor([4./3]).to(dtype).requires_grad_()
    # a = torch.tensor([-0.9312]).to(dtype).requires_grad_()
    # p = torch.tensor([1.077]).to(dtype).requires_grad_()
    eks_model = PseudoLDA(a, p)
    mode = "grad"

    def getloss(a, p, eks_model=None):
        if eks_model is None:
            eks_model = PseudoLDA(a, p)
        loss = 0
        for i,atomz in enumerate(atomzs):
            energy, density = atom(atomz, eks_model=eks_model)
            loss = loss + ((energy - expdata[i]) / expdata[i])**2
        return loss

    if mode == "fwd":
        t0 = time.time()
        for i,atomz in enumerate(atomzs):
            energy, _ = atom(atomz, eks_model=eks_model)
            print("Atom %d: %.5e" % (atomz, energy))
        t1 = time.time()
        print("Forward done in %fs" % (t1-t0))
    elif mode == "grad":
        t0 = time.time()
        loss = getloss(a, p, eks_model)
        t1 = time.time()
        print("Forward done in %fs" % (t1 - t0))
        loss.backward()
        t2 = time.time()
        print("Backward done in %fs" % (t2 - t1))
        agrad = eks_model.a.grad.data
        pgrad = eks_model.p.grad.data

        afd = finite_differences(getloss, (a, p), 0, eps=1e-6, step=1)
        pfd = finite_differences(getloss, (a, p), 1, eps=1e-6, step=1)
        t3 = time.time()
        print("FD done in %fs" % (t3 - t2))

        print("grad of a:")
        print(agrad)
        print(afd)
        print(agrad/afd)

        print("grad of p:")
        print(pgrad)
        print(pfd)
        print(pgrad/pfd)
    elif mode == "opt":
        nsteps = 1000
        opt = torch.optim.SGD(eks_model.parameters(), lr=1e-2)
        for i in range(nsteps):
            opt.zero_grad()
            loss = getloss(a, p, eks_model)
            loss.backward()
            opt.step()
            print("Iter %d: (loss) %.3e (a) %.3e (p) %.3e" % \
                (i, loss.data, eks_model.a.data, eks_model.p.data))
