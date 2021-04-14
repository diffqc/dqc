import torch
import numpy as np
import pytest
from dqc.system.mol import Mol
from dqc.system.sol import Sol
from dqc.hamilton.intor.lattice import Lattice

# these tests to make sure the systems parse the inputs correctly

dtype = torch.float64
moldescs = [
    "H 1.0 0.0 0.0; Be 2.0 0.0 0.0",
    (["H", "Be"], [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
    ([1, 4], torch.tensor([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=dtype)),
]
alattice = torch.eye(3, dtype=dtype) * 4

@pytest.mark.parametrize("moldesc", moldescs)
def test_mol_orbweights(moldesc):
    m = Mol(moldesc, basis="6-311++G**", dtype=dtype)
    orb_weight1 = torch.tensor([2.0, 2.0, 1.0], dtype=dtype)
    assert torch.allclose(m.get_orbweight(), orb_weight1)

    m2 = Mol(moldesc, basis="6-311++G**", dtype=dtype, charge=1)
    orb_weight2 = torch.tensor([2.0, 2.0], dtype=dtype)
    assert torch.allclose(m2.get_orbweight(), orb_weight2)

    m3 = Mol(moldesc, basis="6-311++G**", dtype=dtype, spin=3)
    orb_weight3 = torch.tensor([2.0, 1.0, 1.0, 1.0], dtype=dtype)
    assert torch.allclose(m3.get_orbweight(), orb_weight3)

    # try if it raises an error if spin is invalid
    fail = False
    try:
        m4 = Mol(moldesc, basis="6-311++G**", dtype=dtype, spin=4)
    except AssertionError:
        fail = True
    assert fail

    fail = False
    try:
        m4 = Mol(moldesc, basis="6-311++G**", dtype=dtype, spin=-1)
    except AssertionError:
        fail = True
    assert fail

@pytest.mark.parametrize("moldesc", moldescs)
def test_mol_grid(moldesc):
    # default: level 4
    m = Mol(moldesc, basis="6-311++G**", dtype=dtype)
    m.setup_grid()
    rgrid = m.get_grid().get_rgrid()

    # only check the dimension and the type because the number of grid points
    # can be changed
    assert rgrid.shape[1] == 3
    assert m.get_grid().coord_type == "cart"

def test_mol_nuclei_energy():
    # test the calculation of ion-ion interaction energy (+ gradients w.r.t. pos)
    # mol has fractional feature, so we use that

    def get_ene_ii(atomz, atompos):
        m = Mol((atomz, atompos), basis="6-311++G**", dtype=dtype, spin=1)
        return m.get_nuclei_energy()

    # check the true energy
    atomz = torch.tensor([1.0, 4.0], dtype=dtype).requires_grad_()
    atompos = torch.tensor([[1.0, 0.0, 0.0], [2.5, 0.0, 0.0]], dtype=dtype).requires_grad_()
    ene_ii = get_ene_ii(atomz, atompos)
    true_val = ene_ii * 0 + 4.0 / 1.5
    assert torch.allclose(ene_ii, true_val)

    # check the gradients
    torch.autograd.gradcheck(get_ene_ii, (atomz, atompos))
    torch.autograd.gradgradcheck(get_ene_ii, (atomz, atompos))

def test_mol_pbc_nuclei_energy():
    # test the calculation of ion-ion interaction energy (+ gradients w.r.t. pos)
    # in periodic boundary condition

    def get_ene_ii(atomz, atompos, alattice):
        m = Sol((atomz, atompos), alattice=alattice, basis="6-311++G**", dtype=dtype, spin=1)
        return m.get_nuclei_energy()

    # check the true energy
    atomz = torch.tensor([1], dtype=torch.int32)
    alattice = torch.eye(3, dtype=dtype).requires_grad_()
    atompos = torch.tensor([[0.0, 0.0, 0.0]], dtype=dtype)#.requires_grad_()
    ene_ii = get_ene_ii(atomz, atompos, alattice)

    # check the gradients
    torch.autograd.gradcheck(get_ene_ii, (atomz, atompos, alattice))
    torch.autograd.gradgradcheck(get_ene_ii, (atomz, atompos, alattice))

@pytest.mark.parametrize("moldesc", moldescs)
def test_mol_pbc_orbweights(moldesc):
    m = Sol(moldesc, basis="6-311++G**", alattice=alattice, dtype=dtype)
    orb_weight1 = torch.tensor([2.0, 2.0, 1.0], dtype=dtype)
    assert torch.allclose(m.get_orbweight(), orb_weight1)

    m3 = Sol(moldesc, basis="6-311++G**", alattice=alattice, dtype=dtype, spin=3)
    orb_weight3 = torch.tensor([2.0, 1.0, 1.0, 1.0], dtype=dtype)
    assert torch.allclose(m3.get_orbweight(), orb_weight3)

    # try if it raises an error if spin is invalid
    fail = False
    try:
        m4 = Sol(moldesc, basis="6-311++G**", alattice=alattice, dtype=dtype, spin=4)
    except AssertionError:
        fail = True
    assert fail

    fail = False
    try:
        m4 = Sol(moldesc, basis="6-311++G**", alattice=alattice, dtype=dtype, spin=-1)
    except AssertionError:
        fail = True
    assert fail

@pytest.mark.parametrize("moldesc", moldescs)
def test_mol_pbc_grid(moldesc):
    m = Sol(moldesc, basis="6-311++G**", alattice=alattice, dtype=dtype)
    m.setup_grid()
    rgrid = m.get_grid().get_rgrid()

    # only check the dimension and the type because the number of grid points
    # can be changed
    assert rgrid.shape[1] == 3
    assert m.get_grid().coord_type == "cart"

def test_lattice():
    # testing various properties of the lattice object
    a = torch.tensor([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [2.0, 1.0, 1.0]], dtype=dtype)
    b = torch.inverse(a.T) * (2 * np.pi)
    latt = Lattice(a)
    assert torch.allclose(latt.lattice_vectors(), a)
    assert torch.allclose(latt.recip_vectors(), b)
    assert torch.allclose(latt.volume(), torch.det(a))

    # check the lattice_ls function returns the correct shape
    ls0 = latt.get_lattice_ls(rcut=1.5)  # (nb, ndim)
    assert ls0.ndim == 2
    assert ls0.shape[1] == 3

    # check the ls has no repeated coordinates
    ls0_dist = torch.norm(ls0[:, None, :] - ls0[None, :, :], dim=-1)  # (nb, nb)
    ls0_dist = ls0_dist + torch.eye(ls0_dist.shape[0], dtype=dtype)
    assert torch.all(ls0_dist.abs() > 1e-9)
