import torch
from ddft.modules.base_linear import BaseLinearModule
from ddft.modules.eigen import EigenModule
from ddft.modules.equilibrium import EquilibriumModule
from ddft.utils.misc import set_default_option
from ddft.dft.dft1d import DFT1D

class HamiltonSpatial1D(BaseLinearModule):
    def __init__(self, rgrid):
        super(HamiltonSpatial1D, self).__init__()
        self.rgrid = rgrid # (nr,)
        self.inv_dr = 1.0 / (self.rgrid[1] - self.rgrid[0])
        self.inv_dr2 = self.inv_dr**2
        self._shape = (self.rgrid.shape[0], self.rgrid.shape[0])

    def forward(self, wf, vext):
        # wf: (nbatch, nr) or (nbatch, nr, ncols)
        # vext: (nbatch, nr)

        # normalize the shape of wf
        wfndim = wf.ndim
        if wfndim == 2:
            wf = wf.unsqueeze(-1)

        nbatch = wf.shape[0]
        kinetics = self.kinetics(wf)
        extpot = vext.unsqueeze(-1) * wf
        h = kinetics + extpot

        if wfndim == 2:
            h = h.squeeze(-1)
        return h

    def kinetics(self, wf):
        return (wf - (torch.roll(wf,1,dims=1) + torch.roll(wf,-1,dims=1)) * 0.5) * self.inv_dr2 # (nbatch, nr, ncols)

    @property
    def shape(self):
        return self._shape

    def diag(self, vext):
        return self.inv_dr2 + vext

    def sumwf2(self):
        return self.inv_dr

class VKS1(torch.nn.Module):
    def __init__(self, a, p):
        super(VKS1, self).__init__()
        self.a = torch.nn.Parameter(a)
        self.p = torch.nn.Parameter(p)

    def forward(self, density):
        vks = self.a * density.abs()**self.p
        return vks

def _get_uniform_density(rgrid, focc):
    # rgrid: (nr,)
    # focc: (nbatch, nlowest)
    nbatch = focc.shape[0]
    nr = rgrid.shape[0]

    nels = focc.sum(dim=-1, keepdim=True) # (nbatch, 1)
    dr = rgrid[1] - rgrid[0]
    density_val = nels / dr / nr # (nbatch, 1)
    density = torch.zeros((nbatch, nr)).to(rgrid.dtype).to(rgrid.device) + density_val

    return density

if __name__ == "__main__":
    import time
    import matplotlib.pyplot as plt
    from ddft.utils.fd import finite_differences

    dtype = torch.float64
    nr = 101
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
        "method": "exacteig",
        "verbose": False
    }
    a = torch.tensor([3.0]).to(dtype)
    p = torch.tensor([0.3333]).to(dtype)
    vext = (rgrid * rgrid).unsqueeze(0).requires_grad_() # (nbatch, nr)
    focc = torch.tensor([[2.0, 2.0, 2.0, 1.0]]).requires_grad_() # (nbatch, nlowest)

    def getloss(a, p, vext, focc, return_model=False):
        # set up the modules
        H_model = HamiltonSpatial1D(rgrid)
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

    # # second model
    # a2 = torch.tensor([0.0]).to(dtype)
    # p2 = torch.tensor([0.3333]).to(dtype)
    # vks_model2 = VKS1(a2, p2)
    # dft_model2 = DFTSpatial1D(rgrid, H_model, vks_model2, nlowest)
    # scf_model2 = EquilibriumModule(dft_model2, forward_options=forward_options)
    #
    # density0 = _get_uniform_density(rgrid, focc) # (nbatch, nr)
    # density = scf_model(density0, vext, focc)
    #
    # density2 = scf_model2(density0, vext, focc)
    #
    # plt.plot(density2[0].detach().numpy())
    # plt.plot(density[0].detach().numpy())
    # plt.show()
