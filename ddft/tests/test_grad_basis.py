import torch
from torch.autograd import gradcheck, gradgradcheck
from ddft.basissets.cgto_basis import CGTOBasis
from ddft.grids.radialgrid import LegendreRadialShiftExp
from ddft.grids.sphangulargrid import Lebedev
from ddft.grids.multiatomsgrid import BeckeMultiGrid

dtype = torch.float64

def test_grad_cgto():
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
    # gradgradcheck(fcn, (atomzs, atomposs, wf, vext))
