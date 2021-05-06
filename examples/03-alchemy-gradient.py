import dqc
import torch
import xitorch.optimize

# In this file, we will show how to get the gradient and higher order gradients
# w.r.t. atomic number while keeping the number of electrons constant

dtype = torch.float64
dev = torch.tensor([0.0], dtype=dtype).requires_grad_()

def fcn(atompos, dev):
    atomzs = torch.cat((7. + dev, 7. - dev), dim=0)
    m = dqc.Mol((atomzs, atompos), basis="3-21G", spin=0)
    ene = dqc.HF(m).run().energy()
    return ene

# We should minimize the energy to get the equilibrium position, so the derivative
# we will calculate is the derivative of minimum energy w.r.t. dev, *not* the
# energy at the given position w.r.t. dev.
print("Finding the equilibrium position")
atompos0 = torch.tensor([[1.2, 0.0, 0.0], [-1.2, 0.0, 0.0]], dtype=dtype)
equil_atompos = xitorch.optimize.minimize(fcn, atompos0, (dev,), method="gd",
                                          step=1e-1, maxiter=100, f_rtol=1e-10,
                                          verbose=True)
equil_ene = fcn(equil_atompos, dev)

# gradient of the energy at the equilibrium position w.r.t. dev
deneddev = torch.autograd.grad(equil_ene, dev, create_graph=True)
d2eneddev2 = torch.autograd.grad(deneddev, dev)
print(deneddev, d2eneddev2)
