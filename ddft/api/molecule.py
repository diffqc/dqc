import torch
import numpy as np
from ddft.dft.dft import DFT
from ddft.utils.misc import set_default_option
from ddft.basissets.cgto_basis import CGTOBasis
from ddft.hamiltons.hmolcgauss import HamiltonMoleculeCGauss
from ddft.hamiltons.hmolc0gauss import HamiltonMoleculeC0Gauss
from ddft.grids.radialgrid import LegendreRadialShiftExp
from ddft.grids.sphangulargrid import Lebedev
from ddft.grids.multiatomsgrid import BeckeMultiGrid
from ddft.modules.equilibrium import EquilibriumModule
from ddft.eks import BaseEKS, Hartree, xLDA

__all__ = ["molecule"]

def molecule(atomzs, atompos,
         eks_model="lda",
         basis="6-311++G**",
         # gwmin=1e-5, gwmax=1e2, ng=60,
         rmin=1e-5, rmax=1e2, nr=100,
         angprec=13, lmax_poisson=4,
         dtype=torch.float64, device="cpu",
         eig_options=None, scf_options=None, bck_options=None):

    # atomzs: (natoms,)
    # atompos: (natoms, ndim)

    eig_options = set_default_option({
        "method": "exacteig",
    }, eig_options)
    scf_options = set_default_option({
        "min_eps": 1e-5,
        "jinv0": 0.5,
        "alpha0": 1.0,
        "verbose": True,
        "method": "selfconsistent",
    }, scf_options)
    bck_options = set_default_option({
        "min_eps": 1e-9,
    }, bck_options)

    # normalize the device and eks_model
    device = _normalize_device(device)
    eks_model = _normalize_eks(eks_model)

    # setup the grid
    radgrid = LegendreRadialShiftExp(rmin, rmax, nr, dtype=dtype, device=device)
    sphgrid = Lebedev(radgrid, prec=angprec, basis_maxangmom=lmax_poisson, dtype=dtype, device=device)
    grid = BeckeMultiGrid(sphgrid, atompos, dtype=dtype, device=device)

    # set up the basis
    natoms = atompos.shape[0]
    b = CGTOBasis(basis, cartesian=True, requires_grad=False,
                  dtype=dtype, device=device)
    b.construct_basis(atomzs, atompos)
    H_model = b.get_hamiltonian(grid).to(dtype).to(device)

    # gwidths = torch.logspace(np.log10(gwmin), np.log10(gwmax), ng, dtype=dtype).to(device) # (ng,)
    # alphas = 1./(2*gwidths*gwidths).unsqueeze(-1).repeat(natoms, 1) # (natoms*ng, 1)
    # centres = atompos.unsqueeze(1).repeat_interleave(ng, dim=0) # (natoms*ng, 1, ndim)
    # coeffs = torch.ones_like(alphas, device=device)
    # H_model = HamiltonMoleculeC0Gauss(grid, alphas, centres, coeffs, atompos, atomzs).to(dtype).to(device)

    # set up the occupation number
    nelectrons = int(atomzs.sum())
    nlowest = (nelectrons // 2) + (nelectrons % 2)
    focc = torch.ones(nlowest, dtype=dtype, device=device).unsqueeze(0)
    focc[:,:nelectrons//2] = 2.0

    # run the self-consistent iterations
    dft_model = scf_dft(grid, H_model, focc, eks_model,
        eig_options=eig_options, scf_options=scf_options,
        bck_options=bck_options)
    # get the postprocess' components
    density = dft_model.density()
    energy = dft_model.energy()

    return energy, density

def scf_dft(grid, H_model, focc, eks_model,
        eig_options={}, scf_options={}, bck_options={}):
    # focc: tensor of (nbatch, nlowest)

    dtype, device = H_model.dtype, H_model.device
    nlowest = focc.shape[1]

    hparams = []
    vext = torch.zeros_like(grid.rgrid[:,0]).unsqueeze(0).to(device)

    # setup the modules
    all_eks_models = Hartree()
    if eks_model is not None:
        all_eks_models = all_eks_models + eks_model
    all_eks_models.set_grid(grid)
    dft_model = DFT(H_model, all_eks_models, nlowest, **eig_options)
    scf_model = EquilibriumModule(dft_model, forward_options=scf_options, backward_options=bck_options)

    # calculate the density
    density0 = torch.zeros_like(vext).to(device)
    density0 = dft_model(density0, vext, focc, hparams).detach()
    density = scf_model(density0, vext, focc, hparams)
    return dft_model

# class BasisToEnergy(torch.nn.Module):
#     def __init__(self, grid, eks_model, dtype, device):
#         super(BasisToEnergy, self).__init__()
#         self.grid = grid
#         self.eks_model = eks_model
#         self.dtype = dtype
#         self.device = device
#
#     def forward(self)

def ion_coulomb_energy(atomzs, atompos):
    # atomzs: (natoms,)
    # atompos: (natoms, ndim)
    r12 = (atompos - atompos.unsqueeze(1)).norm(dim=-1) # (natoms, natoms)
    z12 = atomzs * atomzs.unsqueeze(1) # (natoms, natoms)
    infdiag = torch.eye(r12.shape[0], dtype=r12.dtype, device=r12.device)
    idiag = infdiag.diagonal()
    idiag[:] = float("inf")
    r12 = r12 + infdiag
    return (z12 / r12).sum() * 0.5

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

    # setup the molecule's atoms positions
    atomzs = torch.tensor([1.0, 1.0], dtype=dtype)
    distance = torch.tensor([1.0], dtype=dtype).requires_grad_()

    # pseudo-lda eks model
    a = torch.tensor([-0.7385587663820223]).to(dtype).requires_grad_()
    p = torch.tensor([4./3]).to(dtype).requires_grad_()
    eks_model = PseudoLDA(a, p)
    mode = "fwd"

    def getloss(a, p, distance, eks_model=None):
        atompos = distance * torch.tensor([[-0.5], [0.5]], dtype=dtype) # (2,1)
        atompos = torch.cat((atompos, torch.zeros((2,2), dtype=dtype)), dim=-1)
        if eks_model is None:
            eks_model = PseudoLDA(a, p)
        energy, _ = molecule(atomzs, atompos, eks_model=eks_model)
        ion_energy = ion_coulomb_energy(atomzs, atompos)
        loss = (energy+ion_energy).sum()
        return loss

    if mode == "fwd":
        t0 = time.time()
        atompos = torch.tensor([[-distance[0]/2.0, 0.0, 0.0], [distance[0]/2.0, 0.0, 0.0]], dtype=dtype)
        energy, density = molecule(atomzs, atompos, eks_model=eks_model)
        ion_energy = ion_coulomb_energy(atomzs, atompos)
        print("Electron energy: %f" % energy)
        print("Ion energy: %f" % ion_energy)
        print("Total energy: %f" % (ion_energy + energy))
        t1 = time.time()
        print("Forward done in %fs" % (t1-t0))
    elif mode == "grad":
        t0 = time.time()
        loss = getloss(a, p, distance, eks_model)
        t1 = time.time()
        print("Forward done in %fs" % (t1 - t0))
        loss.backward()
        t2 = time.time()
        print("Backward done in %fs" % (t2 - t1))
        agrad = eks_model.a.grad.data
        pgrad = eks_model.p.grad.data
        distgrad = distance.grad.data

        afd = finite_differences(getloss, (a, p, distance), 0, eps=1e-4, step=1)
        pfd = finite_differences(getloss, (a, p, distance), 1, eps=1e-4, step=1)
        distfd = finite_differences(getloss, (a, p, distance), 2, eps=1e-4, step=1)
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

        print("grad of distance:")
        print(distgrad)
        print(distfd)
        print(distgrad/distfd)
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
