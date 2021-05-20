from itertools import product
import numpy as np
import torch
import pytest
import xitorch as xt
from dqc.api.loadbasis import loadbasis
from dqc.qccalc.hf import HF
from dqc.system.mol import Mol
from dqc.system.sol import Sol
from dqc.utils.safeops import safepow, safenorm
from dqc.utils.datastruct import ValGrad

# checks on end-to-end outputs and gradients

dtype = torch.float64
basis = "3-21G"

atomzs_poss = [
    ([1, 1], 1.0),  # "H -0.5 0 0; H 0.5 0 0"
    ([3, 3], 5.0),  # "Li -2.5 0 0; Li 2.5 0 0"
    ([7, 7], 2.0),  # "N -1.0 0 0; N 1.0 0 0"
    ([9, 9], 2.5),  # "F -1.25 0 0; F 1.25 0 0"
    ([6, 8], 2.0),  # "C -1.0 0 0; O 1.0 0 0"
]
energies = [
    # from pyscf with 3-21G basis
    -1.07195346e+00,  # H2
    -1.47683688e+01,  # Li2
    -1.08298897e+02,  # N2
    -1.97636373e+02,  # F2
    -1.12078732e+02,  # CO
]

@pytest.mark.parametrize(
    "atomzs,dist,energy_true,variational",
    [(*atomz_pos, energy, False) for (atomz_pos, energy) in zip(atomzs_poss, energies)] +
    [(*atomz_pos, energy, True) for (atomz_pos, energy) in zip(atomzs_poss, energies)]
)
def test_rhf_energy(atomzs, dist, energy_true, variational):
    # test to see if the energy calculated by DQC agrees with PySCF
    torch.manual_seed(123)

    # only set debugging mode only in one case to save time
    if atomzs == [1, 1]:
        xt.set_debug_mode(True)

    poss = torch.tensor([[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=dtype) * dist
    mol = Mol((atomzs, poss), basis=basis, dtype=dtype)
    qc = HF(mol, restricted=True, variational=variational).run()
    ene = qc.energy()
    assert torch.allclose(ene, ene * 0 + energy_true, rtol=1e-7)

    if atomzs == [1, 1]:
        xt.set_debug_mode(False)

@pytest.mark.parametrize(
    "atomzs,dist,grad2,variational",
    [(*atomz_pos, grad2, varnal) for (atomz_pos, grad2, varnal) in \
        product(atomzs_poss, [False, True], [False, True])]
)
def test_rhf_grad_pos(atomzs, dist, grad2, variational):
    # test grad of energy w.r.t. atom's position

    torch.manual_seed(123)
    # set stringent requirement for grad2
    bck_options = None if (not grad2 and not variational) else {
        "rtol": 1e-10,
        "atol": 1e-10,
    }
    fwd_options = None if not variational else {
        "f_rtol": 1e-15,
        "x_rtol": 1e-15,
        "maxiter": 20000,
    }

    def get_energy(dist_tensor):
        poss_tensor = torch.tensor([[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=dtype) * dist_tensor
        mol = Mol((atomzs, poss_tensor), basis=basis, dtype=dtype)
        qc = HF(mol, restricted=True, variational=variational).run(
            fwd_options=fwd_options, bck_options=bck_options)
        return qc.energy()
    dist_tensor = torch.tensor(dist, dtype=dtype, requires_grad=True)
    if grad2:
        torch.autograd.gradgradcheck(get_energy, (dist_tensor,),
                                     rtol=1e-2, atol=1e-5)
    else:
        if variational:
            torch.autograd.gradcheck(get_energy, (dist_tensor,), rtol=4e-3)
        else:
            torch.autograd.gradcheck(get_energy, (dist_tensor,))

def test_rhf_basis_inputs():
    # test to see if the various basis inputs produce the same results
    atomzs = [1, 1]
    poss = torch.tensor([[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=dtype) * 1.0
    mol0 = Mol((atomzs, poss), basis=basis, dtype=dtype)
    ene0 = HF(mol0, restricted=True).run().energy()

    mol1 = Mol((atomzs, poss), basis=[basis, basis], dtype=dtype)
    ene1 = HF(mol1, restricted=True).run().energy()
    assert torch.allclose(ene0, ene1)

    mol1 = Mol((atomzs, poss), basis=loadbasis("1:" + basis), dtype=dtype)
    ene1 = HF(mol1, restricted=True).run().energy()
    assert torch.allclose(ene0, ene1)

    mol2 = Mol((atomzs, poss), basis={"H": basis}, dtype=dtype)
    ene2 = HF(mol2, restricted=True).run().energy()
    assert torch.allclose(ene0, ene2)

    mol2 = Mol((atomzs, poss), basis={1: basis}, dtype=dtype)
    ene2 = HF(mol2, restricted=True).run().energy()
    assert torch.allclose(ene0, ene2)

    mol2 = Mol((atomzs, poss), basis={1: loadbasis("1:3-21G")}, dtype=dtype)
    ene2 = HF(mol2, restricted=True).run().energy()
    assert torch.allclose(ene0, ene2)

############### Unrestricted Kohn-Sham ###############
u_atomzs_spins = [
    # atomz, spin
    (1, 1),
    (3, 1),
    (5, 1),
    (8, 2),
]
u_atom_energies = [
    -4.96198609e-01,  # H
    -7.38151326e+00,  # Li
    -2.43897617e+01,  # B
    -7.43936572e+01,  # O
]
u_mols_dists_spins = [
    # atomzs,dist,spin
    # ([8, 8], 2.0, 2),  # "O -1.0 0 0; O 1.0 0 0"
    ([7, 8], 2.0, 1),  # "N -1.0 0 0; O 1.0 0 0"
]
u_mols_energies = [
    # -1.48704412e+02,  # O2
    -1.28477807e+02,  # NO
]

@pytest.mark.parametrize(
    "atomzs,dist,energy_true,variational",
    # [(*atomz_pos, energy, True) for (atomz_pos, energy) in zip(atomzs_poss[:2], energies[:2])] + \
    [(*atomz_pos, energy, False) for (atomz_pos, energy) in zip(atomzs_poss[:2], energies[:2])]
)
def test_uhf_energy_same_as_rhf(atomzs, dist, energy_true, variational):
    # test to see if uhf energy gets the same energy as rhf for non-polarized systems
    poss = torch.tensor([[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=dtype) * dist
    mol = Mol((atomzs, poss), basis=basis, dtype=dtype)
    qc = HF(mol, restricted=False, variational=variational).run()
    ene = qc.energy()
    assert torch.allclose(ene, ene * 0 + energy_true, rtol=1e-8)

@pytest.mark.parametrize(
    "atomz,spin,energy_true,variational",
    [(atomz, spin, energy, True) for ((atomz, spin), energy) in zip(u_atomzs_spins, u_atom_energies)] + \
    [(atomz, spin, energy, False) for ((atomz, spin), energy) in zip(u_atomzs_spins, u_atom_energies)]
)
def test_uhf_energy_atoms(atomz, spin, energy_true, variational):
    # check the energy of atoms with non-0 spins
    torch.manual_seed(123)
    poss = torch.tensor([[0.0, 0.0, 0.0]], dtype=dtype)
    mol = Mol(([atomz], poss), basis=basis, dtype=dtype, spin=spin)
    qc = HF(mol, restricted=False, variational=variational).run()
    ene = qc.energy()
    assert torch.allclose(ene, ene * 0 + energy_true, atol=0.0, rtol=1e-7)

@pytest.mark.parametrize(
    "atomzs,dist,spin,energy_true,variational",
    [(atomzs, dist, spin, energy, True) for ((atomzs, dist, spin), energy)
        in zip(u_mols_dists_spins, u_mols_energies)] + \
    [(atomzs, dist, spin, energy, False) for ((atomzs, dist, spin), energy)
        in zip(u_mols_dists_spins, u_mols_energies)]
)
def test_uhf_energy_mols(atomzs, dist, spin, energy_true, variational):
    # check the energy of molecules with non-0 spins
    # NOTE: O2 iteration gets into excited state (probably)
    torch.manual_seed(123)
    poss = torch.tensor([[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=dtype) * dist
    mol = Mol((atomzs, poss), basis=basis, dtype=dtype, spin=spin)
    qc = HF(mol, restricted=False, variational=variational).run()
    ene = qc.energy()
    assert torch.allclose(ene, ene * 0 + energy_true, rtol=1e-8, atol=0.0)

############## Fractional charge ##############
def test_rhf_frac_energy():
    # test if fraction of atomz produces close/same results with integer atomz

    def get_energy(atomz, with_ii=True):
        atomzs = [atomz, atomz]
        poss = torch.tensor([[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=dtype)
        mol = Mol((atomzs, poss), basis=basis, spin=0, dtype=dtype)
        qc = HF(mol, restricted=True).run()
        ene = qc.energy()
        if with_ii:
            ene = ene - mol.get_nuclei_energy()
        return ene

    ene1tot      = get_energy(1, with_ii=True)
    ene1e        = get_energy(1, with_ii=False)
    ene1ftot     = get_energy(1.0, with_ii=True)
    ene1fe       = get_energy(1.0, with_ii=False)
    ene1epse     = get_energy(1.0 + 1e-2, with_ii=False)
    ene1smalltot = get_energy(1.0 + 1e-8, with_ii=True)
    ene1smalle   = get_energy(1.0 + 1e-8, with_ii=False)

    # check if the floating point calculation produces the same number as
    # integer calculation (or close if atomz is close to 1)
    assert torch.allclose(ene1tot, ene1ftot, rtol=0, atol=1e-10)
    assert torch.allclose(ene1e, ene1fe, rtol=0, atol=1e-10)
    assert torch.allclose(ene1tot, ene1smalltot)
    assert torch.allclose(ene1e, ene1smalle)

    # check if the electron energy changes with change of z
    assert torch.all(ene1e != ene1epse)

    # check if the results on the negative side is close to the integer part
    ene2e = get_energy(3, with_ii=False)
    ene2ne = get_energy(3 - 1e-4, with_ii=False)
    assert torch.allclose(ene2e, ene2ne, rtol=3e-4)

def test_rhf_frac_energy_grad():
    # test the gradient of energy w.r.t. Z in fraction case

    def get_energy(atomzs):
        poss = torch.tensor([[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=dtype)
        mol = Mol((atomzs, poss), basis=basis, spin=0, dtype=dtype)
        qc = HF(mol, restricted=True).run()
        ene = qc.energy() - mol.get_nuclei_energy()
        return ene

    atomzs = torch.tensor([1.2, 1.25], dtype=dtype, requires_grad=True)
    torch.autograd.gradcheck(get_energy, (atomzs,))
    torch.autograd.gradgradcheck(get_energy, (atomzs,))
