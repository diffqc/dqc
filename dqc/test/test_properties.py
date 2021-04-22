from typing import Union, List
import torch
import numpy as np
import pytest
import psutil
from dqc.api.properties import hessian_pos, vibration, edipole, equadrupole, \
                               ir_spectrum, raman_spectrum
from dqc.system.mol import Mol
from dqc.qccalc.hf import HF
from dqc.xc.base_xc import BaseXC
from dqc.utils.safeops import safepow
from dqc.utils.datastruct import ValGrad, SpinParam

dtype = torch.float64

@pytest.fixture
def h2o_qc():
    # run the self-consistent HF iteration for h2o
    atomzs = torch.tensor([8, 1, 1], dtype=torch.int64)
    # from CCCBDB (calculated geometry for H2O)
    atomposs = torch.tensor([
        [0.0, 0.0, 0.2156],
        [0.0, 1.4749, -0.8625],
        [0.0, -1.4749, -0.8625],
    ], dtype=dtype).requires_grad_()
    efield = torch.zeros(3, dtype=dtype).requires_grad_()
    grad_efield = torch.zeros((3, 3), dtype=dtype).requires_grad_()

    efields = (efield, grad_efield)
    mol = Mol(moldesc=(atomzs, atomposs), basis="3-21G", dtype=dtype, efield=efields)
    qc = HF(mol).run()
    return qc

def test_hess(h2o_qc):
    # test if the hessian is Hermitian
    hess = hessian_pos(h2o_qc)
    assert torch.allclose(hess, hess.transpose(-2, -1).conj(), atol=2e-6)

def test_vibration(h2o_qc):
    # test if the vibration of h2o is similar to what pyscf computes

    freq_cm1, normcoord = vibration(h2o_qc, freq_unit="cm^-1")

    # from CCCBDB (calculated frequencies for H2O)
    calc_freq_cm1 = torch.tensor([3944., 3811., 1800.], dtype=dtype)

    assert torch.allclose(freq_cm1[:3], calc_freq_cm1, rtol=1e-3)

def test_edipole(h2o_qc):
    # test if the electric dipole of h2o similar to pyscf

    h2o_dip = edipole(h2o_qc, unit="debye")

    # from cccbdb (calculated electric dipole moment for H2O)
    pyscf_h2o_dip = torch.tensor([0, 0, -2.388e+00], dtype=dtype)

    assert torch.allclose(h2o_dip, pyscf_h2o_dip, rtol=2e-4)

def test_equadrupole(h2o_qc):
    # test if the quadrupole properties close to cccbdb precomputed values

    h2o_quad = equadrupole(h2o_qc, unit="debye*angst")

    cccbdb_h2o_quad = torch.tensor([
        [-6.838, 0.0, 0.0],
        [0.0, -3.972, 0.0],
        [0.0, 0.0, -5.882],
    ], dtype=dtype)

    assert torch.allclose(h2o_quad, cccbdb_h2o_quad, rtol=2e-4)

def test_ir_spectrum(h2o_qc):
    freq, ir_ints = ir_spectrum(h2o_qc, freq_unit="cm^-1", ints_unit="km/mol")

    # from CCCBDB (calculated frequencies for H2O)
    calc_freq_cm1 = torch.tensor([3944., 3811., 1800.], dtype=dtype)
    # from CCCBDB (calculated vibrational properties for H2O)
    ir_ints1 = torch.tensor([9.123e+00, 4.7e-02, 7.989e+01], dtype=dtype)

    assert torch.allclose(freq[:3], calc_freq_cm1, rtol=1e-3)
    assert torch.allclose(ir_ints[:3], ir_ints1, rtol=1e-2)

# this test is memory extensive, so skip if there is no available memory
def test_raman_spectrum(h2o_qc):
    freq, raman_ints = raman_spectrum(h2o_qc, freq_unit="cm^-1", ints_unit="angst^4/amu")

    # from CCCBDB (calculated vibrational properties for H2O)
    calc_raman_ints1 = torch.tensor([44.12, 95.71, 11.5], dtype=dtype)
    assert torch.allclose(raman_ints[:3], calc_raman_ints1, rtol=1e-3)

@pytest.mark.parametrize(
    "check_type",
    ["ene", "jac_ene"]
)
def test_properties_gradcheck(check_type):
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

    def get_energy(atomposs, efield, grad_efield):
        efields = (efield, grad_efield)
        mol = Mol(moldesc=(atomzs, atomposs), basis="3-21G", dtype=dtype, efield=efields)
        qc = HF(mol).run()
        ene = qc.energy()
        return ene

    if check_type == "ene":
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

    if check_type == "jac_ene":
        torch.autograd.gradcheck(get_jac_ene, (atomposs.detach(), efield, grad_efield.detach()))
        # raman spectra intensities
        torch.autograd.gradgradcheck(get_jac_ene, (atomposs.detach(), efield, grad_efield.detach()),
                                     atol=3e-4)
