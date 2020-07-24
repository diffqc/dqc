from itertools import product
import torch
import numpy as np
from ddft.basissets.cgto_basis import CGTOBasis
from ddft.hamiltons.hmolcgauss import HamiltonMoleculeCGauss
from ddft.grids.radialgrid import LegendreShiftExpRadGrid
from ddft.grids.sphangulargrid import Lebedev
from ddft.grids.multiatomsgrid import BeckeMultiGrid

# Test procedures for checking the hamiltonian matrix's eigenvalues

dtype = torch.float64

def test_hamilton_molecule_cartesian_gauss():
    def runtest(atomz, nelmts_tensor):
        # setup grid
        atompos = torch.tensor([[0.0, 0.0, 0.0]], dtype=dtype) # (natoms, ndim)
        atomzs = torch.tensor([atomz], dtype=dtype)
        radgrid = LegendreShiftExpRadGrid(200, 1e-6, 1e3, dtype=dtype)
        atomgrid = Lebedev(radgrid, prec=13, basis_maxangmom=4, dtype=dtype)
        grid = BeckeMultiGrid(atomgrid, atompos, dtype=dtype)

        # setup basis
        nbasis = 60
        nelmts_val = 2
        if nelmts_tensor:
            nelmts = torch.ones(nbasis, dtype=torch.int32) * nelmts_val
        else:
            nelmts = nelmts_val
        alphas = torch.logspace(np.log10(1e-4), np.log10(1e6), nbasis*nelmts_val).to(dtype) # (nbasis,)
        centres = atompos.repeat(nbasis*nelmts_val, 1)
        coeffs = torch.ones((nbasis*nelmts_val,))
        ijks = torch.zeros((nbasis*nelmts_val, 3), dtype=torch.int32)
        h = HamiltonMoleculeCGauss(grid, ijks, alphas, centres, coeffs, nelmts, atompos, atomzs).to(dtype)

        # compare the eigenvalues (no degeneracy because the basis is all radial)
        nevals = 5
        evals = get_evals(grid, h)[:nevals]
        true_evals = -0.5*atomz*atomz/(torch.arange(1, nevals+1).to(dtype)*1.0)**2
        print(evals - true_evals)
        assert torch.allclose(evals, true_evals)

    for atomz, nelmts_tensor in product([1.0,2.0], [True, False]):
        runtest(atomz, nelmts_tensor)

def test_hamilton_molecule_cartesian_gauss1():
    def runtest(atomz):
        # setup grid
        atompos = torch.tensor([[0.0, 0.0, 0.0]], dtype=dtype) # (natoms, ndim)
        atomzs = torch.tensor([atomz], dtype=dtype)
        radgrid = LegendreShiftExpRadGrid(200, 1e-6, 1e3, dtype=dtype)
        atomgrid = Lebedev(radgrid, prec=13, basis_maxangmom=4, dtype=dtype)
        grid = BeckeMultiGrid(atomgrid, atompos, dtype=dtype)

        # setup basis
        nbasis = 60
        nelmts = 1
        alphas = torch.logspace(np.log10(1e-4), np.log10(1e6), nbasis).repeat(4).to(dtype) # (4*nbasis,)
        centres = atompos.repeat(4*nbasis, 1)
        coeffs = torch.ones((4*nbasis,))
        ijks = torch.zeros((4,nbasis, 3), dtype=torch.int32)
        # L=1
        ijks[1,:,0] = 1
        ijks[2,:,1] = 1
        ijks[3,:,2] = 1
        ijks = ijks.view(4*nbasis, 3)
        h = HamiltonMoleculeCGauss(grid, ijks, alphas, centres, coeffs, nelmts, atompos, atomzs).to(dtype)

        # compare the eigenvalues (there is degeneracy in p-orbitals)
        nevals = 6
        evals = get_evals(grid, h)[:nevals]
        true_evals = -0.5*atomz*atomz/torch.tensor([1.0, 2.0, 2.0, 2.0, 2.0, 3.0]).to(dtype)**2
        print(evals - true_evals)
        assert torch.allclose(evals, true_evals)

    for atomz in [1.0,2.0]:
        runtest(atomz)

def test_hamilton_molecule_cgto():
    def runtest(atomz, basisname, rtol):
        # setup grid
        atompos = torch.tensor([[0.0, 0.0, 0.0]], dtype=dtype) # (natoms, ndim)
        atomzs = torch.tensor([atomz], dtype=dtype)
        radgrid = LegendreShiftExpRadGrid(200, 1e-6, 1e3, dtype=dtype)
        atomgrid = Lebedev(radgrid, prec=13, basis_maxangmom=4, dtype=dtype)
        grid = BeckeMultiGrid(atomgrid, atompos, dtype=dtype)

        # setup basis
        basis = CGTOBasis(basisname, cartesian=True, dtype=dtype)
        basis.construct_basis(atomzs, atompos)
        h = basis.get_hamiltonian(grid)

        # compare the eigenvalues (there is degeneracy in p-orbitals)
        nevals = 1
        evals = get_evals(grid, h)[:nevals]
        true_evals = -0.5*atomz*atomz/torch.tensor([1.0]).to(dtype)**2
        print(evals - true_evals)
        assert torch.allclose(evals, true_evals, rtol=rtol)

    for atomz in [1.0]:
        runtest(atomz, "6-311++G**", rtol=5e-4)

def get_evals(grid, h, *hparams):
    nr = grid.rgrid.shape[0]
    vext = torch.zeros(1, nr).to(dtype)
    H = h.fullmatrix(vext, *hparams)
    olp = h.overlap.fullmatrix()

    # check symmetricity of those matrices
    assert torch.allclose(olp-olp.transpose(-2,-1), torch.zeros_like(olp))
    assert torch.allclose(H-H.transpose(-2,-1), torch.zeros_like(H))

    mat = torch.solve(H[0], olp[0])[0]
    evals, evecs = torch.eig(mat)
    evals = torch.sort(evals.view(-1))[0]
    return evals
