# This file contains functions that perform DFT in one dimension that are
# differentiable.

import torch
from ddft.hamiltonian import HamiltonNP1D, _apply_G

class DFT1D(torch.nn.Module):
    def __init__(self, model, options=None):
        super(DFT1D, self).__init__()
        self.model = model
        self.options = options

    def forward(self, rgrid, vext, iexc, density0=None):
        # vext: (nbatch, nr)

        density, wf, e, H = _DFT1DForward.apply(
            rgrid, vext, self.model, iexc, density0, self.options)

        # propagate through the backward to construct the meaningful backward
        # graph
        if self.training:
            vks = self.model(density)
            density = _DFT1DBackward.apply(density, wf, e, H,
                rgrid, vext, vks, self.model, self.options)
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
    * H: torch.tensor (nbatch, nr, nr)
        The Hamiltonian of the last iteration.

    Note
    ----
    * The output wf, e, H are only used for backward calculation in the backward
        module. Therefore, the backward of wf, e, and H are not propagated.
    * The output wf, e, H should not be used outside the _DFT1D* modules.
    """
    @staticmethod
    def forward(ctx, rgrid, vext, vks_model, iexc, density0=None, options=None):
        config = _get_default_options(options)
        verbose = config["verbose"]

        # specify the default initial density
        nr = vext.shape[1]
        np = iexc.shape[1]
        length = rgrid.max(dim=-1, keepdim=True)[0] - rgrid.min(dim=-1, keepdim=True)[0] # (nbatch, 1)
        density_val = np / length
        if density0 is None:
            density0 = torch.zeros_like(vext, device=vext.device) + density_val

        # perform the KS iterations
        density = density0
        stop_reason = "max_niter"
        mult = density_val * nr / np
        sqrtmult = mult**0.5
        for i in range(config["max_niter"]):
            # calculate the potential for non-interacting particles
            vks = vks_model(density) # (nbatch, nr)
            vext_tot = vext + vks

            # solving the KS single particle equation
            wf, e, H = HamiltonNP1D.apply(rgrid, vext_tot, iexc,
                True, True) # return wf and hamiltonian
            wf = wf * sqrtmult
            new_density = (wf * wf).sum(dim=-1)

            # check for convergence
            ddensity = (new_density - density).abs().max(dim=-1)[0]
            rddensity = ddensity / density.abs().max(dim=-1)[0] # (nbatch,)
            if verbose:
                print(" Iter %3d reps_dens: %.3e" % (i+1, torch.max(rddensity)))

            # assign the new values for the next iteration
            density = new_density

            # stopping criteria
            if torch.all(rddensity < config["max_reps"]):
                stop_reason = "max_reps"
                break

        # print out the convergence info
        if verbose:
            print("Stopping reason after %d iterations: %s" % (i+1, stop_reason))

        return density, wf, e, H

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
