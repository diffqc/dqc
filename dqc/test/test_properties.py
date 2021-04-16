import torch
import pytest
from dqc.api.properties import hessian_pos, vibration, edipole, equadrupole
from dqc.system.mol import Mol
from dqc.qccalc.ks import KS

dtype = torch.float64

@pytest.fixture
def h2o_qc():
    # run the self-consistent ks-dft iteration for h2o
    atomzs = torch.tensor([8, 1, 1], dtype=torch.int64)
    atomposs = torch.tensor([
        [0.0, 0.0, 0.2217],
        [0.0, 1.4309, -0.8867],
        [0.0, -1.4309, -0.8867],
    ], dtype=dtype).requires_grad_()
    efield = torch.zeros(3, dtype=dtype).requires_grad_()

    mol = Mol(moldesc=(atomzs, atomposs), basis="3-21G", dtype=dtype, efield=efield)
    qc = KS(mol, xc="lda_x+lda_c_pw").run()
    return qc

def test_hess(h2o_qc):
    # test if the hessian is Hermitian
    hess = hessian_pos(h2o_qc)
    assert torch.allclose(hess, hess.transpose(-2, -1).conj(), atol=2e-6)

def atest_vibration(h2o_qc):
    freq, normcoord = vibration(h2o_qc)
    print(freq / 2.4188843265857e-17 / 3e8 / 1e2)
    print(normcoord)
    raise RuntimeError
