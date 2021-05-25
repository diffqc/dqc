import torch
import dqc
atomzs, atomposs = dqc.parse_moldesc("H 1 0 0; H -1 0 0")
atomposs.requires_grad_()
mol = dqc.Mol((atomzs, atomposs), basis="3-21G")
qc = dqc.HF(mol).run()
ene = qc.energy()
force = -torch.autograd.grad(ene, atomposs)[0]
