import torch
import numpy as np
from ddft.modules.base_linear import BaseLinearModule
from ddft.dft.spatial1d import HamiltonSpatial1D
from ddft.modules.eigen import EigenModule
from ddft.modules.equilibrium import EquilibriumModule
from ddft.utils.misc import set_default_option

class HamiltonPW1D(HamiltonSpatial1D):
    def __init__(self, rgrid):
        super(HamiltonPW1D, self).__init__(rgrid)

        # construct the r-grid and q-grid
        N = len(rgrid)
        dr = rgrid[1] - rgrid[0]
        boxsize = rgrid[-1] - rgrid[0]
        dq = 2*np.pi / boxsize
        Nhalf = (N // 2) + 1
        offset = (N + 1) % 2
        qgrid_half = torch.arange(Nhalf)
        self.qgrid = qgrid_half
        # self.qgrid = torch.cat((qgrid_half[1:].flip(0)[offset:], qgrid_half)) # (nr,)
        self.q2 = self.qgrid*self.qgrid

        # initialize the spatial Hamiltonian
        super(HamiltonPW1D, self).__init__(rgrid)

    def kinetics(self, wf):
        # wf: (nbatch, nr, ncols)
        # wf consists of points in the real space

        # perform the operation in q-space, so FT the wf first
        wfT = wf.transpose(-2, -1) # (nbatch, ncols, nr)
        coeff = torch.rfft(wfT, signal_ndim=1) # (nbatch, ncols, nr, 2)

        # multiply with |q|^2 and IFT transform it back
        q2 = self.q2.unsqueeze(-1).expand(-1,2) # (nr, 2)
        coeffq2 = coeff * q2
        kin = torch.irfft(coeffq2, signal_ndim=1) # (nbatch, ncols, nr)

        # revert to the original shape
        return kin.transpose(-2, -1) # (nbatch, nr, ncols)

if __name__ == "__main__":
    import time
    import matplotlib.pyplot as plt
    from ddft.utils.fd import finite_differences
    from ddft.dft.spatial1d import VKS1, _get_uniform_density
    from ddft.dft.dft1d import DFT1D

    dtype = torch.float64
    nr = 101
    boxsize = 4
    max_energy = (nr * np.pi / boxsize)**2 * 0.5
    rgrid = torch.linspace(-2, 2, nr).to(dtype)
    nlowest = 4
    forward_options = {
        "verbose": False,
        "linesearch": False,
    }
    backward_options = {
        "verbose": False
    }
    eigen_options = {
        "method": "davidson",
        "verbose": False
    }
    a = torch.tensor([3.0]).to(dtype)
    p = torch.tensor([0.3333]).to(dtype)
    vext = (rgrid * rgrid).unsqueeze(0).requires_grad_() # (nbatch, nr)
    focc = torch.tensor([[2.0, 2.0, 2.0, 1.0]]).requires_grad_() # (nbatch, nlowest)

    def getloss(a, p, vext, focc, return_model=False):
        # set up the modules
        H_model = HamiltonPW1D(rgrid)
        vks_model = VKS1(a, p)
        dft_model = DFT1D(H_model, vks_model, nlowest,
            **eigen_options)
        scf_model = EquilibriumModule(dft_model,
            forward_options=forward_options,
            backward_options=backward_options)

        # calculate the density
        density0 = _get_uniform_density(rgrid, focc)
        density = scf_model(density0, vext, focc)

        # calculate the defined loss function
        loss = (density*density).sum()
        if not return_model:
            return loss
        else:
            return loss, scf_model

    t0 = time.time()
    loss, scf_model = getloss(a, p, vext, focc, return_model=True)
    t1 = time.time()
    print("Forward done in %fs" % (t1 - t0))
    loss.backward()
    params = list(scf_model.parameters())
    a_grad = params[0].grad.data
    p_grad = params[1].grad.data
    vext_grad = vext.grad.data
    focc_grad = focc.grad.data
    t2 = time.time()
    print("Backward done in %fs" % (t2 - t1))

    # use finite_differences
    a_fd = finite_differences(getloss, (a, p, vext, focc), 0, eps=1e-6)
    p_fd = finite_differences(getloss, (a, p, vext, focc), 1, eps=1e-6)
    vext_fd = finite_differences(getloss, (a, p, vext, focc), 2, eps=1e-3)
    focc_fd = finite_differences(getloss, (a, p, vext, focc), 3, eps=1e-5)
    t3 = time.time()
    print("Finite differences done in %fs" % (t3 - t2))

    print("a gradients:")
    print(a_grad)
    print(a_fd)
    print(a_grad / a_fd)

    print("p gradients:")
    print(p_grad)
    print(p_fd)
    print(p_grad / p_fd)

    print("vext gradients:")
    print(vext_grad)
    print(vext_fd)
    print(vext_grad / vext_fd)

    print("focc gradients:")
    print(focc_grad)
    print(focc_fd)
    print(focc_grad / focc_fd)
