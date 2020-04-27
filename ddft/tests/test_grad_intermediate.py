import torch
from torch.autograd import gradcheck, gradgradcheck
from ddft.eks.base_eks import BaseEKS
from ddft.eks.hartree import Hartree
from ddft.dft.dft import DFT
from ddft.basissets.cgto_basis import CGTOBasis
from ddft.grids.radialgrid import LegendreRadialShiftExp
from ddft.grids.sphangulargrid import Lebedev
from ddft.grids.multiatomsgrid import BeckeMultiGrid
from ddft.utils.safeops import safepow

"""
Test the gradient for the intermediate methods (not basic module and not API)
"""

dtype = torch.float64

def test_grad_basis_cgto():
    basisname = "6-311++G**"
    ns0 = 7

    rmin = 1e-5
    rmax = 1e2
    nr = 100
    prec = 13
    nrgrid = 148 * nr

    def fcn(atomzs, atomposs, wf, vext):
        radgrid = LegendreRadialShiftExp(rmin, rmax, nr, dtype=dtype)
        sphgrid = Lebedev(radgrid, prec=prec, basis_maxangmom=4, dtype=dtype)
        grid = BeckeMultiGrid(sphgrid, atomposs, dtype=dtype)
        basis = CGTOBasis(basisname, cartesian=True)
        basis.construct_basis(atomzs, atomposs, requires_grad=False)
        H_model = basis.get_hamiltonian(grid)
        # print(H_model.rgrid.shape)
        # print(H_model.shape)
        y = H_model(wf, vext)
        return (y**2).sum()

    atomzs = torch.tensor([1.0, 1.0], dtype=dtype)
    atomposs = torch.tensor([[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=dtype).requires_grad_()
    ns = ns0 * len(atomzs)
    wf = torch.ones((1,ns,1), dtype=dtype)
    vext = torch.zeros((1,nrgrid), dtype=dtype)

    gradcheck(fcn, (atomzs, atomposs, wf, vext))
    gradgradcheck(fcn, (atomzs, atomposs, wf, vext))

def test_grad_dft_cgto():
    basisname = "6-311++G**"
    rmin = 1e-5
    rmax = 1e2
    nr = 100
    prec = 13

    class PseudoLDA(BaseEKS):
        def __init__(self, a, p):
            super(PseudoLDA, self).__init__()
            self.a = a
            self.p = p

        def forward(self, density):
            return self.a * safepow(density.abs(), self.p)

    def fcn(atomzs, atomposs, a, p, output="energy"):
        radgrid = LegendreRadialShiftExp(rmin, rmax, nr, dtype=dtype)
        sphgrid = Lebedev(radgrid, prec=prec, basis_maxangmom=4, dtype=dtype)
        grid = BeckeMultiGrid(sphgrid, atomposs, dtype=dtype)
        basis = CGTOBasis(basisname, cartesian=True, dtype=dtype)
        basis.construct_basis(atomzs, atomposs, requires_grad=False)
        H_model = basis.get_hamiltonian(grid)

        focc = torch.tensor([[2.0, 0.0]], dtype=dtype)
        nlowest = focc.shape[1]
        vext = torch.zeros_like(grid.rgrid[:,0]).unsqueeze(0)

        all_eks_models = Hartree() + PseudoLDA(a, p)
        all_eks_models.set_grid(grid)

        eig_options = {"method": "exacteig"}
        dft_model = DFT(H_model, all_eks_models, nlowest, **eig_options)

        # set up the dft model
        dft_model.set_vext(vext)
        dft_model.set_focc(focc)
        dft_model.set_hparams([])

        # get the density and energy
        density0 = torch.zeros_like(vext)
        density = dft_model(density0)
        energy = dft_model.energy(density)
        if output == "energy":
            return energy
        elif output == "density":
            return density.abs().sum()

    a = torch.tensor([-0.7385587663820223]).to(dtype).requires_grad_()
    p = torch.tensor([4./3]).to(dtype).requires_grad_()
    atomzs = torch.tensor([1.0, 1.0], dtype=dtype)
    atomposs = torch.tensor([[-0.5, 0.1, 0.1], [0.5, 0.1, 0.1]], dtype=dtype).requires_grad_()

    gradcheck(fcn, (atomzs, atomposs, a, p, "energy"))
    gradcheck(fcn, (atomzs, atomposs, a, p, "density"))
    # choosing smaller eps make the numerical method produces nan, don't know why
    gradgradcheck(fcn, (atomzs, atomposs, a, p, "energy"), eps=1e-3)
    gradgradcheck(fcn, (atomzs, atomposs, a, p, "density"), eps=1e-3)
