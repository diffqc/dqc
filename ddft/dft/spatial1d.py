import torch
from ddft.transforms.base_transform import SymmetricTransform
from ddft.utils.tensor import roll_1
from ddft.utils.rootfinder import lbfgs, broyden
from ddft.utils.eigpairs import davidson

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

        def loss(density):
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

            # calculate the density and normalize so that the sum is equal to density
            new_density = (wf * wf).sum(dim=-1) # (nbatch, nr)
            new_density = new_density / new_density.sum() * density.sum()

            return density - new_density

        # perform the rootfinder
        density = lbfgs(loss, density0, max_niter=40, verbose=True)

        return density

    @staticmethod
    def backward(ctx, grad_density, grad_wf, grad_e, grad_H):
        return (None, None, None, None, None, None)

class _DFT1DBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, density, wf, e, H, rgrid, vext, vks, model_vks, options=None):
        config = _get_default_options(options)
        ctx.config = config

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

        density_temp = density.detach().clone().requires_grad_()
        with torch.enable_grad():
            vks_temp = model_vks(density_temp) # (nbatch, nr)

        def _N(grad):
            # grad will be (nbatch, nr)
            res, = torch.autograd.grad(vks_temp, (density_temp,),
                grad_outputs=grad,
                retain_graph=True)
            return res

        def _G(grad):
            return _apply_G(wf, e, H, grad_wf_or_n=grad, is_grad_wf=False)

        grad_accum = _G(grad_density)
        dgrad = grad_accum
        for i in range(config["max_niter"]):
            # apply N and G then accummulate the gradient
            dgrad = _G(_N(dgrad)) # (nbatch, nr)
            grad_accum = grad_accum + dgrad

            # calculate the relative residual of dgrad
            resid = dgrad.abs().max(dim=-1)[0]
            grad_mag = grad_accum.abs().max(dim=-1)[0]
            rresid = resid / grad_mag
            if verbose:
                print("Iter %3d: %.3e | %.3e | %.3e" % \
                     (i+1, resid, grad_mag, rresid))

            # check the stopping condition
            if torch.all(rresid < config["max_reps"]):
                break

        grad_vext = grad_accum
        grad_vks = grad_accum

        return (grad_density,
            None, None, None, # wf, e, H
            None, # rgrid
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

if __name__ == "__main__":
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

    a = torch.tensor([4.4]).to(dtype).requires_grad_()
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
