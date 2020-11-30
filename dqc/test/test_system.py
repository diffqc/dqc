import torch
import pytest
from dqc.system.mol import Mol

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
