import torch
import dqc
import time

# set up the H2 molecule, forcing the atom positions to be differentiable
mol = dqc.Mol("H -1 0 0; H 1 0 0", basis="3-21G", diffparams=["atompos"])
qc = dqc.HF(mol).run()
ene = qc.energy()  # calculate the energy
force = -torch.autograd.grad(ene, mol.atompos)[0]  # calculate the force

print(force)
