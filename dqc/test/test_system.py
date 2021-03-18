import torch
import numpy as np
import pytest
from dqc.system.mol import Mol
from dqc.system.tools import Lattice

# these tests to make sure the systems parse the inputs correctly

dtype = torch.float64
moldescs = [
    "H 1.0 0.0 0.0; Be 2.0 0.0 0.0",
    (["H", "Be"], [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
    ([1, 4], torch.tensor([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=dtype)),
]

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

def test_lattice():
    # testing various properties of the lattice object
    a = torch.tensor([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [2.0, 1.0, 1.0]], dtype=dtype)
    b = torch.inverse(a.T) * (2 * np.pi)
    latt = Lattice(a)
    assert torch.allclose(latt.lattice_vectors(), a)
    assert torch.allclose(latt.recip_vectors(), b)
    assert torch.allclose(latt.volume(), torch.det(a))

    # check the lattice_ls function returns the correct shape
    nimgs = 2
    ls0 = latt.get_lattice_ls(nimgs=nimgs)  # (nb, ndim)
    assert ls0.ndim == 2
    assert ls0.shape[0] == (2 * nimgs + 1) ** 3
    assert ls0.shape[1] == 3

    # check the ls has no repeated coordinates
    ls0_dist = torch.norm(ls0[:, None, :] - ls0[None, :, :], dim=-1)  # (nb, nb)
    ls0_dist = ls0_dist + torch.eye(ls0_dist.shape[0], dtype=dtype)
    assert torch.all(ls0_dist.abs() > 1e-9)
