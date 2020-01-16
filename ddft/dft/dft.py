import torch
from ddft.modules.eigen import EigenModule
from ddft.utils.misc import set_default_option

class DFT(torch.nn.Module):
    def __init__(self, H_model, vks_model, nlowest, **eigen_options):
        super(DFT, self).__init__()
        eigen_options = set_default_option({
            "v_init": "randn",
        }, eigen_options)
        self.H_model = H_model
        self.vks_model = vks_model
        self.eigen_model = EigenModule(H_model, nlowest, **eigen_options)

    def forward(self, density, vext, focc):
        # density: (nbatch, nr)
        # vext: (nbatch, nr)
        # focc: (nbatch, nlowest)

        # calculate the total potential experienced by Kohn-Sham particles
        vks = self.vks_model(density) # (nbatch, nr)
        vext_tot = vext + vks

        # compute the eigenpairs
        # evals: (nbatch, nlowest), evecs: (nbatch, nr, nlowest)
        eigvals, eigvecs = self.eigen_model(vext_tot)

        # normalize the norm of density
        eigvec_dens = (eigvecs*eigvecs) # (nbatch, nr, nlowest)
        eigvec_dens = self.H_model.getdens(eigvec_dens)
        dens = eigvec_dens * focc.unsqueeze(1) # (nbatch, nr, nlowest)
        new_density = dens.sum(dim=-1) # (nbatch, nr)

        return new_density

def _get_uniform_density(rgrid, nels):
    # rgrid: (nr,)
    # nels: (nbatch,)
    nbatch = nels.shape[0]
    nr = rgrid.shape[0]

    nels = nels.unsqueeze(-1) # (nbatch, 1)
    dr = rgrid[1] - rgrid[0]
    density_val = nels / dr / nr # (nbatch, 1)
    density = torch.zeros((nbatch, nr)).to(rgrid.dtype).to(rgrid.device) + density_val

    return density

if __name__ == "__main__":
    import time
    import matplotlib.pyplot as plt
    from ddft.utils.fd import finite_differences
    from ddft.hamiltons.hspatial1d import HamiltonSpatial1D
    from ddft.hamiltons.hpw1d import HamiltonPW1D
    from ddft.modules.equilibrium import EquilibriumModule

    class VKS1(torch.nn.Module):
        def __init__(self, a, p):
            super(VKS1, self).__init__()
            self.a = torch.nn.Parameter(a)
            self.p = torch.nn.Parameter(p)

        def forward(self, density):
            vks = self.a * density.abs()**self.p
            return vks

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
        # H_model = HamiltonSpatial1D(rgrid)
        H_model = HamiltonPW1D(rgrid)
        vks_model = VKS1(a, p)
        dft_model = DFT(H_model, vks_model, nlowest,
            **eigen_options)
        scf_model = EquilibriumModule(dft_model,
            forward_options=forward_options,
            backward_options=backward_options)

        # calculate the density
        nels = focc.sum(-1)
        density0 = _get_uniform_density(rgrid, nels)
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
