import torch
import pytest
from dqc.system.mol import Mol

dtype = torch.float64

@pytest.fixture
def system1():
    poss = torch.tensor([[0.0, 0.0, 0.8], [0.0, 0.0, -0.8]], dtype=dtype)
    moldesc = ([1, 1], poss)
    m = Mol(moldesc, basis="6-311++G**", dtype=dtype)
    m.setup_grid()
    hamilton = m.get_hamiltonian()
    hamilton.build()
    hamilton.setup_grid(m.get_grid())
    return m

def test_cgto_ao2dm(system1):
    # test ao2dm with a simple case

    hamilton1 = system1.get_hamiltonian()
    nao = hamilton1.get_kinnucl().shape[-1]
    norb = nao // 2
    nbatch_orbw = (3,)
    nbatch_orb  = (2, 1)
    nbatch_res  = (2, 3)
    w = 2.0

    # prepare the orbital
    orb = torch.zeros((*nbatch_orb, nao, norb), dtype=dtype)
    orb_diag = orb.diagonal(dim1=-2, dim2=-1)
    orb_diag[:] = 1.0

    # prepare the orbital weights
    orb_weight = torch.ones((*nbatch_orbw, norb), dtype=dtype) * w  # (*BW, norb)

    # calculate the dm and the true dm
    dm = hamilton1.ao_orb2dm(orb, orb_weight)  # (*BOW, nao, nao)
    dm_true = torch.zeros((*nbatch_res, nao, nao), dtype=dtype)
    dm_true_diag = dm_true.diagonal(dim1=-2, dim2=-1)
    dm_true_diag[..., :norb] = w

    assert tuple(dm.shape) == (*nbatch_res, nao, nao)
    assert torch.allclose(dm, dm_true)

def test_cgto_dm2grid(system1):
    # test the function aodm2dens in hamilton cgto

    torch.manual_seed(123)
    hamilton1 = system1.get_hamiltonian()
    nao = hamilton1.get_kinnucl().shape[-1]
    nb = 4

    # prepare the density matrix
    # 0: zeros
    # 1: half of 2nd row (random symmetric)
    # 2: double the 1st row (random symmetric)
    # 3: eye
    dm = torch.rand((nb, nao, nao), dtype=dtype)
    dm = dm + dm.transpose(-2, -1)  # (nb, nao, nao)
    dm[0] = 0
    dm[2] = 2 * dm[1]
    dm[3] = 0
    dm3_diag = dm[3].diagonal(dim1=-2, dim2=-1)
    dm3_diag[:] = 1.0

    # prepare the grids
    # 0th row: middle point between two atoms
    # 1st row: position of the first atom
    # 2nd row: position of the second atom
    grids = torch.tensor([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.4],
        [0.0, 0.0, 0.8],
        [0.0, 0.0, -0.4],
        [0.0, 0.0, -0.8]
    ], dtype=dtype).unsqueeze(1)  # (ngrid, 1, ndim)
    ngrid = grids.shape[0]

    # # pyscf code to obtain the actual results
    # from pyscf import gto
    # from pyscf.dft import numint
    # mol = gto.M(atom="H 0 0 0.8; H 0 0 -0.8", basis="6-311++G**", unit="Bohr")
    # ao = numint.eval_ao(mol, grids[:, 0, :].numpy())
    # dens1 = numint.eval_rho(mol, ao, dm[1].numpy())
    # dens2 = numint.eval_rho(mol, ao, dm[2].numpy())
    # dens3 = numint.eval_rho(mol, ao, dm[3].numpy())
    # print(dens1)
    # print(dens2)
    # print(dens3)
    # raise RuntimeError

    dens = hamilton1.aodm2dens(dm, grids)  # (ngrid, nb)
    dens_true = torch.tensor([
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [3.25841833, 4.37242451, 8.11849071, 4.0871612, 6.25558526],
        [6.51683666, 8.74484901, 16.23698143, 8.1743224, 12.51117051],
        [0.87780038, 1.05290457, 2.14278122, 1.05290457, 2.14278122],
    ], dtype=dens.dtype).transpose(-2, -1)
    assert list(dens.shape) == [ngrid, nb]
    assert torch.allclose(dens, dens_true)

def test_cgto_vext(system1):
    # test the external potential integration
    hamilton1 = system1.get_hamiltonian()
    nao = hamilton1.get_kinnucl().shape[-1]
    ngrid = system1.get_grid().get_rgrid().shape[-2]

    # uniform potential to see the integration of the basis
    w = 3.0
    vext = torch.ones((ngrid, ), dtype=dtype) * w
    a = hamilton1.get_vext(vext).fullmatrix()
    assert torch.allclose(torch.diagonal(a), torch.tensor(w, dtype=dtype))
