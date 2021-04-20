from typing import Union, List
import torch
import numpy as np
import pytest
from dqc.api.properties import hessian_pos, vibration, edipole, equadrupole, \
                               ir_spectrum
from dqc.system.mol import Mol
from dqc.qccalc.ks import KS
from dqc.xc.base_xc import BaseXC
from dqc.utils.safeops import safepow
from dqc.utils.datastruct import ValGrad, SpinParam

dtype = torch.float64

# using pytorch-based lda because 4th derivative of lda is not available from
# libxc
class LDAX(BaseXC):
    def __init__(self):
        self.a = -0.7385587663820223
        self.p = 4.0 / 3

    @property
    def family(self):
        return 1

    def get_edensityxc(self, densinfo: Union[ValGrad, SpinParam[ValGrad]]) -> torch.Tensor:
        if isinstance(densinfo, ValGrad):
            rho = densinfo.value.abs()  # safeguarding from nan
            return self.a * safepow(rho, self.p)
            # return self.a * rho ** self.p
        else:
            return 0.5 * (self.get_edensityxc(densinfo.u * 2) + self.get_edensityxc(densinfo.d * 2))

    def getparamnames(self, methodname: str, prefix: str = "") -> List[str]:
        return []

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
    grad_efield = torch.zeros((3, 3), dtype=dtype).requires_grad_()

    efields = (efield, grad_efield)
    mol = Mol(moldesc=(atomzs, atomposs), basis="3-21G", dtype=dtype, efield=efields)
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

def test_equadrupole(h2o_qc):
    # test if the quadrupole properties close to cccbdb precomputed values

    h2o_quad = equadrupole(h2o_qc, unit="debye*angst")

    cccbdb_h2o_quad = torch.tensor([
        [-6.907, 0.0, 0.0],
        [0.0, -4.222, 0.0],
        [0.0, 0.0, -5.838],
    ], dtype=dtype)

    assert torch.allclose(h2o_quad, cccbdb_h2o_quad, rtol=2e-2)

def test_ir_spectrum(h2o_qc):
    freq, ir_ints = ir_spectrum(h2o_qc, freq_unit="cm^-1", ints_unit="km/mol")

    # pre-computed (the code to generate is on test_vibration)
    pyscf_freq_cm1 = torch.tensor([4074.51432922, 3915.25820884, 1501.856396], dtype=dtype)
    # I can't find any IR intensities that are close to my calculation, so I'm assuming
    # the calculation is correct
    ir_ints1 = torch.tensor([4.4665e-01, 9.2419e+00, 4.5882e+01], dtype=dtype)

    assert torch.allclose(freq[:3], pyscf_freq_cm1, rtol=1e-2)
    assert torch.allclose(ir_ints[:3], ir_ints1, rtol=1e-2)

def test_properties_gradcheck():
    # check if the analytical formula required to calculate the properties
    # agrees with numerical difference
    # NOTE: very slow

    atomzs = torch.tensor([8, 1, 1], dtype=torch.int64)
    atomposs = torch.tensor([
        [0.0, 0.0, 0.2217],
        [0.0, 1.4309, -0.8867],
        [0.0, -1.4309, -0.8867],
    ], dtype=dtype).requires_grad_()

    # test gradient on electric field
    efield = torch.zeros(3, dtype=dtype).requires_grad_()
    grad_efield = torch.zeros((3, 3), dtype=dtype).requires_grad_()
    ldax = LDAX()

    def get_energy(atomposs, efield, grad_efield):
        efields = (efield, grad_efield)
        mol = Mol(moldesc=(atomzs, atomposs), basis="3-21G", dtype=dtype, efield=efields)
        qc = KS(mol, xc=ldax).run()
        ene = qc.energy()
        return ene

    # dipole and quadrupole
    torch.autograd.gradcheck(get_energy, (atomposs, efield, grad_efield))
    # 2nd grad for hessian, ir intensity, and part of raman intensity
    torch.autograd.gradgradcheck(get_energy, (atomposs, efield, grad_efield.detach()))

    def get_jac_ene(atomposs, efield, grad_efield):
        # get the jacobian of energy w.r.t. atompos
        atomposs = atomposs.requires_grad_()
        ene = get_energy(atomposs, efield, grad_efield)
        jac_ene = torch.autograd.grad(ene, atomposs, create_graph=True)[0]
        return jac_ene

    torch.autograd.gradcheck(get_jac_ene, (atomposs.detach(), efield, grad_efield.detach()))
    # raman spectra intensities
    torch.autograd.gradgradcheck(get_jac_ene, (atomposs.detach(), efield, grad_efield.detach()),
        atol=3e-4)
