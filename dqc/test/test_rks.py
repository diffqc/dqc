import torch
from dqc.qccalc.rks import RKS
from dqc.system.mol import Mol

dtype = torch.float64

def test_rks():
    moldesc = "H -0.5 0 0; H 0.5 0 0"
    mol = Mol(moldesc, basis="6-311++G**", dtype=dtype)
    qc = RKS(mol, xc="lda,").run()
    print(qc.energy())
    raise RuntimeError()
