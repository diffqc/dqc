import time
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import gradcheck, gradgradcheck
from ddft.eks import BaseEKS
from ddft.utils.safeops import safepow
from ddft.systems import mol
from ddft.qccalcs import dft

"""
Test various configurations using the molecule API.
"""

dtype = torch.float64
class PseudoLDA(BaseEKS):
    def __init__(self, a, p):
        super(PseudoLDA, self).__init__()
        self.a = a
        self.p = p

    def forward(self, density):
        return self.a * safepow(density.abs(), self.p)

    def getfwdparams(self):
        return [self.a, self.p]

    def setfwdparams(self, *params):
        self.a, self.p = params[:2]
        return 2

def get_atom(atomname):
    basis = "6-311++G**"
    if atomname == "He":
        atomz = 2.0
        energy = -2.90 # ???
        # pyscf_energy = -2.72102435e # LDA, 6-311++G**, grid level 4
    elif atomname == "Be":
        atomz = 4.0
        energy = -14.2219
        # pyscf_energy = -14.2207456807576 # LDA, 6-311++G**, grid level 4
    elif atomname == "Ne":
        atomz = 10.0
        energy = -127.4718
        # pyscf_energy = -127.469035524253 # LDA, 6-311++G**, grid level 4
    atomzs = torch.tensor([atomz], dtype=dtype)
    energy = torch.tensor(energy, dtype=dtype)
    atomposs = torch.tensor([[0.0, 0.0, 0.0]], dtype=dtype)
    return atomzs, atomposs, basis, energy

def test_atom2():
    systems = {
        "Be 0 0 0": -14.2219,
        "Ne 0 0 0": -127.4718
    }
    basis = "6-311++G**"
    runtest_molsystem_energy(systems, basis)

def test_mol2():
    # NOTE: The effect of increasing nr in radial grid has significant effect on
    # Li2. Investigate!
    systems = {
        "H -0.5 0 0; H 0.5 0 0"  : -0.9791401, # pyscf: -0.979143262 # LDA, 6-311++G**, grid level 4
        "Li -2.5 0 0; Li 2.5 0 0": -14.393459, # pyscf: -14.3927863482007 # LDA, 6-311++G**, grid level 4
        "N -1 0 0; N 1 0 0"      : -107.7327, # pyscf: -107.726124017789 # LDA, 6-311++G**, grid level 4
        "F -1.25 0 0; F 1.25 0 0": -197.0101, # pyscf: -197.005308558326 # LDA, 6-311++G**, grid level 4
        "C -1 0 0; O 1 0 0"      : -111.49737, # pyscf: -111.490687028797 # LDA, 6-311++G**, grid level 4
    }
    basis = "6-311++G**"
    runtest_molsystem_energy(systems, basis)

def runtest_molsystem_energy(systems, basis):
    for s in systems:
        energy_true = systems[s]
        m = mol(s, basis)
        scf = dft(m, eks_model="lda")
        energy = scf.energy()
        print("%.7f" % energy)
        assert torch.allclose(energy, torch.tensor(energy_true, dtype=energy.dtype))

if __name__ == "__main__":
    test_mol2()
