from itertools import product
import numpy as np
import torch
import pytest
from dqc.qccalc.uks import UKS
from dqc.qccalc.rks import RKS
from dqc.system.mol import Mol
from dqc.xc.base_xc import BaseXC
from dqc.utils.datastruct import ValGrad
from dqc.utils.safeops import safepow, safenorm

dtype = torch.float64

atomzs_poss_nonpol = [
    ([1, 1], 1.0),
    ([3, 3], 5.0),
]
energies_nonpol = [
    -0.979143260,  # pyscf: -0.979143262
    -14.3927863482007,  # pyscf: -14.3927863482007
]

@pytest.mark.parametrize(
    "xc,atomzs,dist,energy_true",
    [("lda,", *atomz_pos, energy) for (atomz_pos, energy) in zip(atomzs_poss_nonpol, energies_nonpol)]
)
def test_uks_energy_same_as_rks(xc, atomzs, dist, energy_true):
    # test to see if uks energy gets the same energy as rks for non-polarized systems
    poss = torch.tensor([[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=dtype) * dist
    mol = Mol((atomzs, poss), basis="6-311++G**", dtype=dtype)
    qc = UKS(mol, xc=xc).run()
    ene = qc.energy()
    assert torch.allclose(ene, ene * 0 + energy_true)
