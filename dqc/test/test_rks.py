import torch
import pytest
from dqc.qccalc.rks import RKS
from dqc.system.mol import Mol

dtype = torch.float64

@pytest.mark.parametrize(
    "xc,moldesc,energy_true",
    [
        ("lda,", "H -0.5 0 0; H 0.5 0 0", -0.979143260),  # pyscf: -0.979143262
        # ("lda,", "Li -2.5 0 0; Li 2.5 0 0", -14.393459),  # pyscf: -14.3927863482007
        # ("lda,", "N -1 0 0; N 1 0 0", -107.7327),  # pyscf: -107.726124017789
        # ("lda,", "F -1.25 0 0; F 1.25 0 0", -197.0101),  # pyscf: -197.005308558326
        # ("lda,", "C -1 0 0; O 1 0 0", -111.49737),  # pyscf: -111.490687028797
    ]
)
def test_rks_energy(xc, moldesc, energy_true):
    mol = Mol(moldesc, basis="6-311++G**", dtype=dtype)
    qc = RKS(mol, xc=xc).run()
    ene = qc.energy()
    assert torch.allclose(ene, ene * 0 + energy_true)

if __name__ == "__main__":
    xc = "lda,"
    basis = "6-311++G**"
    # mol = Mol("Li -2.5 0 0; Li 2.5 0 0", basis="6-311++G**", dtype=dtype)
    mol = Mol("H -0.5 0 0; H 0.5 0 0", basis=basis, dtype=dtype)
    qc = RKS(mol, xc=xc)#.run()
    # ene = qc.energy()
    # print(ene)
