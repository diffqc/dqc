import torch
from ddft.transforms.base_transform import SymmetricTransform
from ddft.transforms.ntransform import NTransform
from ddft.utils.tensor import roll_1
from ddft.utils.rootfinder import lbfgs, broyden
from ddft.utils.eigpairs import davidson
from ddft.utils.orthogonalize import orthogonalize

class DFT1D(torch.nn.Module):
    def __init__(self, model, options=None):
        super(DFT1D, self).__init__()
        self.model = model
        self.options = options

    def forward(self, rgrid, vext, iexc, density0=None):
        # vext: (nbatch, nr)

        density, wf, e = _DFT1DForward.apply(
            rgrid, vext, self.model, iexc, density0, self.options)

        # propagate through the backward to construct the meaningful backward
        # graph
        if self.training:
            vks = self.model(density)
            density = _DFT1DBackward.apply(density, wf, e,
                rgrid, iexc, vext, vks, self.model, self.options)
        return density

class _DFT1DForward(torch.autograd.Function):
    """
    Solves DFT in one dimension by constructing the explicit Hamiltonian matrix.
    This module should be paired with the backward module for easier backward
    calculation (because we only need the final point of the self-consistent
    iteration).

    Arguments
    ---------
    * rgrid: torch.tensor (nbatch, nr)
        The spatial grid of the space. It is assumed that it has the regular
        spacing.
    * vext: torch.tensor (nbatch, nr)
        The external potential experienced by all particles.
    * vks_model: torch.nn.Module
        Callable model that takes the electron density as input and produces
        the Kohn-Sham potentials (i.e. ee-interactions, kinetics, and
        exchange-correlations).
    * iexc: torch.tensor (nbatch, np)
        The indices of Kohn-Sham particles to be involved in the calculation.
    * density0: torch.tensor (nbatch, nr)
        The initial guess of the density. If unspecified, then it will be
        uniform with the number of particles.
    * options: dict or None
        Dictionary that specifies the option of the solver.
        Leave it to None to use the default solver.

    Returns
    -------
    * density: torch.tensor (nbatch, nr)
        The total electron density.
    * wf: torch.tensor (nbatch, nr, np)
        The wavefunction of Kohn-Sham particles.
    * e: torch.tensor (nbatch, np)
        The energy of Kohn-Sham particles.

    Note
    ----
    * The output wf, e are only used for backward calculation in the backward
        module. Therefore, the backward of wf, e are not propagated.
    * The output wf, e should not be used outside the _DFT1D* modules.
    """
    @staticmethod
    def forward(ctx, rgrid, vext, vks_model, iexc, density0=None, options=None):
        config = _get_default_options(options)
        verbose = config["verbose"]

        # specify the default initial density
        nb, nr = vext.shape
        np = iexc.shape[1]
        if density0 is None:
            length = rgrid.max(dim=-1, keepdim=True)[0] - rgrid.min(dim=-1, keepdim=True)[0] # (nbatch, 1)
            density_val = np / length
            density0 = torch.zeros_like(vext, device=vext.device) + density_val

        # perform the KS iterations
        stop_reason = "max_niter"
        neig = iexc.max() + 1

        def loss(density, return_all=False):
            # calculate the potential for non-interacting particles
            vks = vks_model(density) # (nbatch, nr)
            vext_tot = vext + vks

            # solving the KS single particle equation
            Htransform = _Hamiltonian1D(rgrid, vext_tot, iexc)

            # obtain `neig` eigenpairs with the lowest eigenvalues
            # eigvals: (nbatch, neig)
            # eigvecs: (nbatch, nr, neig)
            eigvals, eigvecs = davidson(Htransform, neig)

            # access the needed eigenpairs
            e = torch.gather(eigvals, dim=1, index=iexc) # (nbatch, np)
            iexc_expand = iexc.unsqueeze(1).expand(-1, nr, -1)
            wf = torch.gather(eigvecs, dim=2, index=iexc_expand) # (nbatch, nr, np)

            mult = (density.sum(-1) / (wf*wf).sum(dim=-1).sum(dim=-1))**0.5 # (nbatch)
            wf = wf * mult.unsqueeze(-1).unsqueeze(-1)

            # calculate the density and normalize so that the sum is equal to density
            new_density = (wf * wf).sum(dim=-1) # (nbatch, nr)
            new_density = new_density / new_density.sum() * density.sum()

            if return_all:
                return density, wf, e
            else:
                return density - new_density

        # perform the rootfinder
        density = lbfgs(loss, density0, max_niter=40, verbose=False)
        density, wf, e = loss(density, return_all=True)

        return density, wf, e

    @staticmethod
    def backward(ctx, grad_density, grad_wf, grad_e):
        return (None, None, None, None, None, None)

class _DFT1DBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, density, wf, e, rgrid, iexc, vext, vks, model_vks, options=None):
        config = _get_default_options(options)
        ctx.config = config
        H = _Hamiltonian1D(rgrid, vext + vks, iexc)

        # save for backward purposes
        ctx.density = density
        ctx.wf = wf
        ctx.e = e
        ctx.H = H
        ctx.rgrid = rgrid
        ctx.vext = vext
        ctx.vks = vks
        ctx.model_vks = model_vks
        return density

    @staticmethod
    def backward(ctx, grad_density):
        # pull out the information from forward
        density = ctx.density # (nbatch, nr)
        wf = ctx.wf # (nbatch, nr, np)
        e = ctx.e # (nbatch, np)
        H = ctx.H # (nbatch, nr, nr)
        model_vks = ctx.model_vks
        config = ctx.config
        verbose = config["verbose"]

        # prepare the transformation
        N = NTransform(model_vks, density)
        G = _G1D(wf, e, H)
        back_transform = -G * (N*G - 1.0).inv()

        # calculate the gradient
        grad_v = back_transform(grad_density)
        grad_vext = grad_v
        grad_vks = grad_v

        return (grad_density,
            None, None, # wf, e
            None, None, # rgrid, iexc
            grad_vext, grad_vks,
            None, None # model_vks, options
            )

def _get_default_options(options):
    # set up the options
    config = {
        "max_niter": 1000,
        "verbose": False,
        "max_reps": 1e-9,
    }
    if options is None:
        options = {}
    config.update(options)
    return config

class _Hamiltonian1D(SymmetricTransform):
    def __init__(self, rgrid, vext, iexc):
        # rgrid: (nbatch, nr)
        # vext: (nbatch, nr)
        # iexc: (nbatch, np)
        self.rgrid = rgrid
        self.vext = vext
        self.iexc = iexc
        self.max_iexc = self.iexc.max()
        self.dr = (rgrid[:,1] - rgrid[:,0]).unsqueeze(-1)
        self.inv_dr2 = 1.0 / (self.dr * self.dr)

        # construct the shape and the diagonal
        self._device = self.vext.device
        self._shape = (self.vext.shape[0], self.vext.shape[1], self.vext.shape[1])

    def _forward(self, wf):
        # wf: (nbatch, nr)

        # hamiltonian: kinetics + vext
        kinetics = (wf - 0.5 * (roll_1(wf, 1) + roll_1(wf, -1))) * self.inv_dr2 # (nbatch, nr)
        extpot = wf * self.vext
        return kinetics + extpot

    def diag(self):
        if not hasattr(self, "_diag"):
            self._diag = self.vext + torch.ones_like(self.vext).to(self._device)
        return self._diag

    @property
    def shape(self):
        return self._shape

    @property
    def dtype(self):
        return self.vext.dtype

class _G1D(SymmetricTransform):
    """
    (dg / dv)^T where g is n_out.
    (dg / dv) = sum_b [2 * diag(wf_b) * A_b^{-1} * diag(wf_b)]
    """
    def __init__(self, wf, e, H):
        # wf: (nbatch, nr, nparticles)
        # e: (nbatch, nparticles)
        # H: transformation of (nbatch, nr, nr)
        self.wf = wf
        self.e = e
        self.H = H

        # obtain the transformation for each particle
        self.nparticles = e.shape[-1]
        self.As = [(-(H - e[:,b])) for b in range(self.nparticles)]
        self.Ainvs = [(-(H - e[:,b])).inv() for b in range(self.nparticles)]

    def _forward(self, x):
        # x is (nbatch, nr)
        wfxs = self.wf * x.unsqueeze(-1) # (nbatch, nr, nparticles)
        wfxs = orthogonalize(wfxs, self.wf, dim=1) # (nbatch, nr, nparticles)
        ys = None
        for b in range(self.nparticles):
            wfb = self.wf[:,:,b]
            wfx = wfxs[:,:,b]
            Abx = self.Ainvs[b](wfx) # (nbatch, nr)
            Abx = orthogonalize(Abx, wfb, dim=-1)

            y = wfb * Abx

            if ys is None:
                ys = y
            else:
                ys = ys + y
        return 2 * ys

    @property
    def shape(self):
        return self.H.shape

    @property
    def dtype(self):
        return self.H.dtype

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from ddft.utils.fd import finite_differences

    class VKSSimpleModel(torch.nn.Module):
        def __init__(self, a, p):
            super(VKSSimpleModel, self).__init__()
            self.a = torch.nn.Parameter(a)
            self.p = torch.nn.Parameter(p)

        def forward(self, density):
            return self.a * density**self.p

    # set up
    dtype = torch.float64
    inttype = torch.long

    def getloss(rgrid, vext, iexc, a, p):
        vks_model = VKSSimpleModel(a, p)
        return getloss2(rgrid, vext, iexc, vks_model)

    def getloss2(rgrid, vext, iexc, vks_model):
        dft1d = DFT1D(vks_model)
        density = dft1d(rgrid.unsqueeze(0), vext.unsqueeze(0), iexc.unsqueeze(0))
        return (density**4).sum()


    rgrid = torch.linspace(-2, 2, 101).to(dtype)
    vext = (rgrid * rgrid).requires_grad_()
    iexc = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 4]).to(inttype)
    a = torch.tensor([1.0]).to(dtype).requires_grad_()
    p = torch.tensor([0.3]).to(dtype).requires_grad_()

    # calculate gradient with backprop
    vks_model = VKSSimpleModel(a, p)
    loss = getloss2(rgrid, vext, iexc, vks_model)
    loss.backward()
    vext_grad = vext.grad.data
    a_grad = vks_model.a.grad.data
    p_grad = vks_model.p.grad.data

    # calculate gradient with finite_differences
    vext_fd = finite_differences(getloss, (rgrid, vext, iexc, a, p), 1,
                eps=1e-4)
    a_fd = finite_differences(getloss, (rgrid, vext, iexc, a, p), 3,
                eps=1e-4)
    p_fd = finite_differences(getloss, (rgrid, vext, iexc, a, p), 4,
                eps=1e-4)
    print("Grad of vext:")
    print(vext_grad)
    print(vext_fd)
    print(vext_grad / vext_fd)

    print("Grad of a:")
    print(a_grad)
    print(a_fd)
    print(a_grad / a_fd)

    print("Grad of p:")
    print(p_grad)
    print(p_fd)
    print(p_grad / p_fd)
    raise RuntimeError



    import matplotlib.pyplot as plt

    class VKSSimpleModel(torch.nn.Module):
        def __init__(self, a, p):
            super(VKSSimpleModel, self).__init__()
            self.a = torch.nn.Parameter(a)
            self.p = torch.nn.Parameter(p)

        def forward(self, density):
            return self.a * density**self.p

    inttype = torch.long
    dtype = torch.float64

    rgrid = torch.linspace(-2, 2, 101).to(dtype)
    vext = (rgrid * rgrid).requires_grad_()
    iexc = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 4]).to(inttype)

    a = torch.tensor([1.0]).to(dtype).requires_grad_()
    p = torch.tensor([0.3]).to(dtype).requires_grad_()
    vks_model = VKSSimpleModel(a, p)
    density = _DFT1DForward.apply(rgrid.unsqueeze(0),
        vext.unsqueeze(0), vks_model, iexc.unsqueeze(0))

    a = torch.tensor([0.0]).to(dtype).requires_grad_()
    p = torch.tensor([0.3]).to(dtype).requires_grad_()
    vks_model = VKSSimpleModel(a, p)
    density2 = _DFT1DForward.apply(rgrid.unsqueeze(0),
        vext.unsqueeze(0), vks_model, iexc.unsqueeze(0))

    plt.plot(density2[0].detach().numpy())
    plt.plot(density[0].detach().numpy())
    plt.show()
