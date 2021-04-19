import torch
import numpy as np
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

def test_vibration(h2o_qc):
    # test if the vibration of h2o is similar to what pyscf computes

    freq_cm1, normcoord = vibration(h2o_qc, freq_unit="cm^-1")

    # pre-computed (the code to generate is below)
    pyscf_freq_cm1 = torch.tensor([4074.51432922, 3915.25820884, 1501.856396], dtype=dtype)

    # # code to generate the frequencies above
    # from pyscf import gto, dft
    # from pyscf.prop.freq import rks
    # mol = gto.M(atom='''
    #             O 0 0 0.2217
    #             H 0  1.4309 -0.8867
    #             H 0 -1.4309 -0.8867''',
    #             basis='321g', unit="Bohr")
    # mf = dft.RKS(mol, xc="lda_x,lda_c_pw").run()
    # w, modes = rks.Freq(mf).kernel()

    # NOTE: rtol is a bit high, init?
    assert torch.allclose(freq_cm1[:3], pyscf_freq_cm1, rtol=1e-2)

def test_edipole(h2o_qc):
    # test if the electric dipole of h2o similar to pyscf

    h2o_dip = edipole(h2o_qc, unit="debye")

    # precomputed dipole moment from pyscf (code to generate is below)
    pyscf_h2o_dip = torch.tensor([-7.35382039e-16, -9.80612124e-15, -2.31439912e+00], dtype=dtype)

    # # code to generate the dipole moment
    # from pyscf import gto, dft
    # from pyscf.prop.freq import rks
    # mol = gto.M(atom='''
    #             O 0 0 0.2217
    #             H 0  1.4309 -0.8867
    #             H 0 -1.4309 -0.8867''',
    #             basis='321g', unit="Bohr")
    # mf = dft.RKS(mol, xc="lda_x,lda_c_pw").run()
    # mf.dip_moment()

    assert torch.allclose(h2o_dip, pyscf_h2o_dip, rtol=3e-4)

def atest_equadrupole(h2o_qc):
    h2o_quad = equadrupole(h2o_qc, unit="da")
    print(h2o_quad)
    raise RuntimeError()
