import torch
import dqc
import xitorch.optimize

# This example shows how to get the equilibrium positions using DQC, xitorch, and pytorch

def get_ene(atompos: torch.Tensor) -> torch.Tensor:
    atomzs = ["H", "H"]  # H2
    mol = dqc.Mol((atomzs, atompos), basis="3-21G")
    qc = dqc.HF(mol).run()
    ene = qc.energy()  # calculate the energy
    return ene

atompos0 = torch.tensor([[1, 0, 0], [-1, 0, 0]], dtype=torch.float64)
minpos = xitorch.optimize.minimize(get_ene, atompos0, method="gd", step=0.8, verbose=True)

print("Equilibrium position:")
print(minpos)
