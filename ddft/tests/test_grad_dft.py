import torch
from torch.autograd import gradcheck, gradgradcheck
from ddft.dft.dft import DFT
from ddft.eks.base_eks import BaseEKS
from ddft.eks.hartree import Hartree
from ddft.basissets.cgto_basis import CGTOBasis
from ddft.grids.radialgrid import LegendreRadialShiftExp
from ddft.grids.sphangulargrid import Lebedev
from ddft.grids.multiatomsgrid import BeckeMultiGrid
from ddft.utils.safeops import safepow

dtype = torch.float64

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

    def fcn(atomzs, atomposs, a, p):
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
        density0 = torch.zeros_like(vext)
        density = dft_model(density0, vext, focc, [])
        return density.abs().sum()

    a = torch.tensor([-0.7385587663820223]).to(dtype).requires_grad_()
    p = torch.tensor([4./3]).to(dtype).requires_grad_()
    atomzs = torch.tensor([1.0, 1.0], dtype=dtype)
    atomposs = torch.tensor([[-0.5, 0.1, 0.1], [0.5, 0.1, 0.1]], dtype=dtype).requires_grad_()

    gradcheck(fcn, (atomzs, atomposs, a, p))
    gradgradcheck(fcn, (atomzs, atomposs, a, p), eps=1e-3)
