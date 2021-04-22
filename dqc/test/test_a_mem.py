import torch
import pytest
from dqc.qccalc.ks import KS
from dqc.qccalc.hf import HF
from dqc.system.mol import Mol
from dqc.xc.base_xc import BaseXC
from dqc.utils.safeops import safepow
from dqc.utils.datastruct import ValGrad
from dqc.test.utils import assert_no_memleak_tensor

# checks on memory leaks in Kohn-Sham iterations

# This file is renamed to "test_a_mem.py" to make sure this file is executed
# first before everything else.
# Running other tests first with memory leak will increase the memory leak
# consumption, so the small changes due to this test is undetectable.

dtype = torch.float64

class PseudoLDA(BaseXC):
    def __init__(self, a, p):
        self.a = a
        self.p = p

    @property
    def family(self):
        return 1

    def get_edensityxc(self, densinfo):
        if isinstance(densinfo, ValGrad):
            rho = densinfo.value.abs()  # safeguarding from nan
            return self.a * safepow(rho, self.p)
        else:
            return 0.5 * (self.get_edensityxc(densinfo.u * 2) + self.get_edensityxc(densinfo.d * 2))

    # the default get_vxc was the source of memory leak!
    # so we don't rewrite it here to test the memleak in the default get_vxc

    def getparamnames(self, methodname, prefix=""):
        return [prefix + "a", prefix + "p"]

ks_mols_dists_spins = [
    # atomzs,dist,spin
    ([1, 1], 1.0, 0),
]

############### Memleak test ###############
@pytest.mark.parametrize(
    "atomzs,dist,spin",
    ks_mols_dists_spins
)
def test_ks_mols_mem_nn(atomzs, dist, spin):
    # test if there's a leak if using neural network xc
    def _test_ks_mols():
        # setting up xc
        a = torch.tensor(-0.7385587663820223, dtype=dtype, requires_grad=True)
        p = torch.tensor(1.3333333333333333, dtype=dtype, requires_grad=True)
        xc = PseudoLDA(a=a, p=p)

        poss = torch.tensor([[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=dtype) * dist
        poss = poss.requires_grad_()
        mol = Mol((atomzs, poss), basis="6-311++G**", grid=3, dtype=dtype, spin=spin)
        qc = KS(mol, xc=xc).run()
        ene = qc.energy()

        # see if grad and backward create memleak
        grads = torch.autograd.grad(ene, (poss, a, p), create_graph=True)
        ene.backward()  # no create_graph here because of known pytorch's leak

    assert_no_memleak_tensor(_test_ks_mols)

@pytest.mark.parametrize(
    "atomzs,dist,spin",
    ks_mols_dists_spins
)
def test_hf_mols_mem_nn(atomzs, dist, spin):
    # test if there's a leak if using neural network xc
    def _test_hf_mols():
        poss = torch.tensor([[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=dtype) * dist
        poss = poss.requires_grad_()
        mol = Mol((atomzs, poss), basis="6-311++G**", dtype=dtype, spin=spin)
        qc = HF(mol).run()
        ene = qc.energy()

        # see if grad and backward create memleak
        grads = torch.autograd.grad(ene, (poss,), create_graph=True)
        ene.backward()  # no create_graph here because of known pytorch's leak

    assert_no_memleak_tensor(_test_hf_mols)
