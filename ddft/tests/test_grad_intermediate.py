import torch
from torch.autograd import gradcheck, gradgradcheck
from ddft.eks.base_eks import BaseEKS
from ddft.eks.hartree import Hartree
from ddft.basissets.cartesian_cgto import CartCGTOBasis
from ddft.grids.radialgrid import LegendreShiftExpRadGrid, LegendreLogM3RadGrid
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
        radgrid = LegendreShiftExpRadGrid(nr, rmin, rmax, dtype=dtype)
        sphgrid = Lebedev(radgrid, prec=prec, basis_maxangmom=4, dtype=dtype)
        grid = BeckeMultiGrid(sphgrid, atomposs, dtype=dtype)
        bases_list = [CartCGTOBasis(atomz, basisname, dtype=dtype) for atomz in atomzs]
        h = bases_list[0].construct_hamiltonian(grid, bases_list, atomposs)
        H_model = h.get_hamiltonian(vext)
        y = H_model.mm(wf)
        return (y**2).sum()

    atomzs = torch.tensor([1.0, 1.0], dtype=dtype)
    atomposs = torch.tensor([[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=dtype).requires_grad_()
    ns = ns0 * len(atomzs)
    wf = torch.ones((1,ns,1), dtype=dtype)
    vext = torch.zeros((1,nrgrid), dtype=dtype)

    gradcheck(fcn, (atomzs, atomposs, wf, vext))
    gradgradcheck(fcn, (atomzs, atomposs, wf, vext))

def test_grad_poisson_radial():
    radgrid = LegendreLogM3RadGrid(nr=100, ra=2.)
    r = radgrid.rgrid.squeeze(-1) # (nr,)
    w = torch.linspace(0.8, 1.2, 5, dtype=r.dtype, device=r.device).unsqueeze(-1) # (nw,1)
    w = w.requires_grad_()
    f = torch.exp(-r/w) # (nw, nr)

    # analytically calculated gradients
    fpois_true = w*w*f + 2*w*w*w/r*torch.expm1(-r/w) # (nw, nr)
    gwidth_true = 4*w*f.mean(dim=-1, keepdim=True) + (f*r).mean(dim=-1, keepdim=True) +\
        6*w*w*(torch.expm1(-r/w)/r).mean(dim=-1, keepdim=True) # (nw,1)

    fpois = radgrid.solve_poisson(f) # (nw, nr)
    loss = fpois.mean(dim=-1).sum()
    gwidth, = torch.autograd.grad(loss, (w,), retain_graph=True)
    assert torch.allclose(gwidth, gwidth_true)

    print(gwidth.view(-1))
    print(gwidth_true.view(-1))
    print((gwidth_true-gwidth).view(-1))

def test_grad_spherical_radial():
    radgrid = LegendreLogM3RadGrid(nr=100, ra=2.)
    grid = Lebedev(radgrid, prec=13, basis_maxangmom=4)
    r = grid.rgrid[:,0] # (nr)
    w = torch.linspace(0.8, 1.2, 5, dtype=r.dtype, device=r.device).unsqueeze(-1) # (nw,1)
    w = w.requires_grad_()
    f = torch.exp(-r/w) # (nw, nr)

    # analytically calculated gradients
    fpois_true = w*w*f + 2*w*w*w/r*torch.expm1(-r/w) # (nw, nr)
    gwidth_true = 4*w*f.mean(dim=-1, keepdim=True) + (f*r).mean(dim=-1, keepdim=True) +\
        6*w*w*(torch.expm1(-r/w)/r).mean(dim=-1, keepdim=True) # (nw,1)

    fpois = grid.solve_poisson(f) # (nw, nr)
    loss = fpois.mean(dim=-1).sum()
    gwidth, = torch.autograd.grad(loss, (w,), retain_graph=True)
    assert torch.allclose(gwidth, gwidth_true)

    print(gwidth.view(-1))
    print(gwidth_true.view(-1))
    print((gwidth_true-gwidth).view(-1))
