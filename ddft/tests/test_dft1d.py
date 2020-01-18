import time
import torch
import matplotlib.pyplot as plt
from ddft.utils.fd import finite_differences
from ddft.hamiltons.hspatial1d import HamiltonSpatial1D
from ddft.hamiltons.hamiltonpw import HamiltonPlaneWave
from ddft.modules.equilibrium import EquilibriumModule
from ddft.spaces.qspace import QSpace
from ddft.dft.dft import DFT, _get_uniform_density

# slow
def test_dft1d_1():
    class EKS1(torch.nn.Module):
        def __init__(self, a, p):
            super(EKS1, self).__init__()
            self.a = torch.nn.Parameter(a)
            self.p = torch.nn.Parameter(p)

        def forward(self, density):
            vks = self.a * density.abs()**self.p
            return vks

    dtype = torch.float64
    nr = 101
    rgrid = torch.linspace(-5, 5, nr).to(dtype)
    nlowest = 4
    forward_options = {
        "verbose": False,
        "linesearch": False,
    }
    backward_options = {
        "verbose": False
    }
    eigen_options = {
        "method": "exacteig",
        "verbose": False
    }
    a = torch.tensor([1.0]).to(dtype)
    p = torch.tensor([1.3333]).to(dtype)
    vext = (rgrid * rgrid * 0.5).unsqueeze(0).requires_grad_() # (nbatch, nr)
    focc = torch.tensor([[2.0, 2.0, 2.0, 1.0]]).requires_grad_() # (nbatch, nlowest)

    def getloss(a, p, vext, focc, return_model=False):
        # set up the modules
        qspace = QSpace(rgrid.unsqueeze(-1), (len(rgrid),))
        H_model = HamiltonPlaneWave(qspace)
        eks_model = EKS1(a, p)
        dft_model = DFT(H_model, eks_model, nlowest,
            **eigen_options)
        scf_model = EquilibriumModule(dft_model,
            forward_options=forward_options,
            backward_options=backward_options)

        # calculate the density
        nels = focc.sum(-1)
        density0 = _get_uniform_density(rgrid, nels)
        density = scf_model(density0, vext, focc)
        energy = dft_model.energy(density, vext, focc)

        # calculate the defined loss function
        loss = (density*density).sum() + (energy*energy).sum()
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
    assert torch.allclose(a_grad/a_fd, torch.ones_like(a_grad), rtol=1e-4)

    print("p gradients:")
    print(p_grad)
    print(p_fd)
    print(p_grad / p_fd)
    assert torch.allclose(p_grad/p_fd, torch.ones_like(p_grad), rtol=1e-4)

    print("vext gradients:")
    print(vext_grad)
    print(vext_fd)
    print(vext_grad / vext_fd)
    assert torch.allclose(vext_grad/vext_fd, torch.ones_like(vext_grad), rtol=1e-2)

    print("focc gradients:")
    print(focc_grad)
    print(focc_fd)
    print(focc_grad / focc_fd)
    assert torch.allclose(focc_grad/focc_fd, torch.ones_like(focc_grad), rtol=1e-2)
