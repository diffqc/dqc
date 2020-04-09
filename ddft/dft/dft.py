import functools
import torch
import numpy as np
from ddft.modules.eigen import EigenModule
from ddft.utils.misc import set_default_option, unpack
from ddft.utils.safeops import safepow
from ddft.eks import VKS

class DFT(torch.nn.Module):
    """
    Perform one forward pass of the DFT self-consistent field approach.

    Class arguments
    ---------------
    * H_model: BaseHamilton
        Hamiltonian transformation.
    * eks_model: torch.nn.Module
        Model that calculates the energy density of the Kohn-Sham energy, given
        the density. The total Kohn-Sham energy is
        integral(eks_model(density) * dr)
    * nlowest: int
        Indicates how many Kohn-Sham particles to be retrieved with the lowest
        energies during the diagonalization process.
    * **eigen_options: kwargs
        The options to be passed to the diagonalization algorithm.

    Forward arguments
    -----------------
    * density: torch.tensor (nbatch, nr)
        The density in the spatial grid.
    * vext: torch.tensor (nbatch, nr)
        The external potential (excluding the Kohn-Sham potential).
    * focc: torch.tensor (nbatch, nlowest)
        The occupation factor of the Kohn-Sham orbitals.
    * hparams: list of torch.tensor (nbatch, ...)
        List of parameters in the Hamiltonian forward method other than `vext`.
    * rparams: list of torch.tensor (nbatch, ...)
        List of parameters used for the overlap operator in the Hamiltonian.

    Forward returns
    ---------------
    * new_density: torch.tensor (nbatch, nr)
        The new density calculated by one forward pass of the self-consistent
        field calculation.
    """
    def __init__(self, H_model, eks_model, nlowest, **eigen_options):
        super(DFT, self).__init__()
        self.H_model = H_model
        self.eks_model = eks_model
        self.vks_model = VKS(eks_model, H_model.grid)
        self.eigen_model = EigenModule(H_model, nlowest,
            rlinmodule=H_model.overlap, **eigen_options)

    def forward(self, density, vext, focc, *params):
        # density: (nbatch, nr)
        # vext: (nbatch, nr)
        # focc: (nbatch, nlowest)

        # unpack the parameters
        hparams, rparams = unpack(params, [self.H_model.nhparams, self.H_model.nolp_params])

        # calculate the total potential experienced by Kohn-Sham particles
        vks = self.vks_model(density) # (nbatch, nr)
        vext_tot = vext + vks

        # compute the eigenpairs
        # evals: (nbatch, nlowest), evecs: (nbatch, nr, nlowest)
        eigvals, eigvecs = self.eigen_model((vext_tot, *hparams), rparams=rparams)

        # normalize the norm of density
        eigvec_dens = self.H_model.getdens(eigvecs) # (nbatch, nr, nlowest)
        dens = eigvec_dens * focc.unsqueeze(1) # (nbatch, nr, nlowest)
        new_density = dens.sum(dim=-1) # (nbatch, nr)

        # save variables for the post-process calculations
        self._lc_vks = vks
        self._lc_vext_tot = vext_tot
        self._lc_eigvals = eigvals
        self._lc_eigvecs = eigvecs
        self._lc_density = density
        self._lc_focc = focc
        # parameters that would be computed later
        self._lc_energy = None

        return new_density

    ############################# post processing #############################
    def density(self):
        return self._lc_density

    def energy(self):
        # calculate the total potential experienced by Kohn-Sham particles
        # from the last forward calculation
        if self._lc_energy is not None:
            return self._lc_energy

        vks = self._lc_vks
        vext_tot = self._lc_vext_tot
        eigvals = self._lc_eigvals
        eigvecs = self._lc_eigvecs
        density = self._lc_density
        focc = self._lc_focc

        # calculates the Kohn-Sham energy
        eks_density = self.eks_model(density) # energy density (nbatch, nr)
        Eks = self.H_model.grid.integralbox(eks_density, dim=-1) # (nbatch,)

        # calculate the individual non-interacting particles energy
        sum_eigvals = (eigvals * focc).sum(dim=-1) # (nbatch,)
        vks_integral = self.H_model.grid.integralbox(vks*density, dim=-1)

        # compute the interacting particles energy
        Etot = sum_eigvals - vks_integral + Eks
        self._lc_energy = Etot
        return Etot

class DFTMulti(torch.nn.Module):
    """
    Perform one forward pass of the DFT self-consistent field approach using
    multiple type of Hamiltonian.
    Multiple type of Hamiltonian is usually used in radial system where the
    kinetics term is different for different angular momentums.
    The density is the sum of all orbitals from the list of Hamiltonian.

    Class arguments
    ---------------
    * H_models: list of BaseHamilton
        List of Hamiltonian transformation.
    * eks_model: torch.nn.Module
        Model that calculates the energy density of the Kohn-Sham energy, given
        the density. The total Kohn-Sham energy is
        integral(eks_model(density) * dr)
    * nlowests: list of int
        Indicates how many Kohn-Sham particles to be retrieved with the lowest
        energies during the diagonalization process.
    * **eigen_options: kwargs
        The options to be passed to the diagonalization algorithm.

    Forward arguments
    -----------------
    * density: torch.tensor (nbatch, nr)
        The density in the spatial grid.
    * vext: torch.tensor (nbatch, nr)
        The external potential (excluding the Kohn-Sham potential).
    * focc: list torch.tensor (nbatch, nlowest)
        The occupation factor of the Kohn-Sham orbitals.
    * all_hparams: list of list of torch.tensor (nbatch, ...)
        List of parameters in the Hamiltonian forward method other than `vext`.
    * all_rparams: list of list of torch.tensor (nbatch, ...)
        List of parameters used for the overlap operator in the Hamiltonian.

    Forward returns
    ---------------
    * new_density: torch.tensor (nbatch, nr)
        The new density calculated by one forward pass of the self-consistent
        field calculation.
    """
    def __init__(self, H_models, eks_model, nlowests, **eigen_options):
        super(DFTMulti, self).__init__()
        self.H_models = H_models
        self.eks_model = eks_model
        self.grid = self.H_models[0].grid
        self.vks_model = VKS(eks_model, self.grid)
        self.eigen_models = [EigenModule(H_model, nlowest,
            rlinmodule=H_model.overlap, **eigen_options) \
            for (H_model, nlowest) in zip(self.H_models, nlowests)]

    def forward(self, density, vext, foccs, *params):
        # density: (nbatch, nr)
        # vext: (nbatch, nr)
        # foccs: list of (nbatch, nlowest)

        # unpack the parameters
        nhparams = [H.nhparams for H in self.H_models]
        nrparams = [H.nolp_params for H in self.H_models]
        all_hparams, all_rparams = unpack(params, [nhparams, nrparams])

        # calculate the total potential experienced by Kohn-Sham particles
        vks = self.vks_model(density) # (nbatch, nr)
        vext_tot = vext + vks

        # compute the eigenpairs
        # all_evals: list of (nbatch, nlowest)
        # all_evecs: list of (nbatch, nr, nlowest)
        if all_rparams == []:
            all_rparams = [[] for _ in range(len(self.H_models))]
        rs = [eigen_model((vext_tot, *hparams), rparams=rparams) \
              for (eigen_model, hparams, rparams) \
              in zip(self.eigen_models, all_hparams, all_rparams)]
        all_eigvals = [r[0] for r in rs]
        all_eigvecs = [r[1] for r in rs]

        # normalize the norm of density
        # dens: list of (nbatch, nr)
        dens = [(H_model.getdens(eigvecs) * focc.unsqueeze(1)).sum(dim=-1) \
            for (H_model, eigvecs, focc)\
            in zip(self.H_models, all_eigvecs, foccs)] # (nbatch, nr, nlowest)
        new_density = functools.reduce(lambda x,y: x + y, dens)

        # save variables for the post-process calculations
        self._lc_vks = vks
        self._lc_vext_tot = vext_tot
        self._lc_all_eigvals = all_eigvals
        self._lc_all_eigvecs = all_eigvecs
        self._lc_density = density
        self._lc_foccs = foccs
        # parameters that would be computed later
        self._lc_energy = None

        return new_density

    ############################# post processing #############################
    def density(self):
        return self._lc_density

    def energy(self):
        # calculate the total potential experienced by Kohn-Sham particles
        # from the last forward calculation
        if self._lc_energy is not None:
            return self._lc_energy

        vks = self._lc_vks
        vext_tot = self._lc_vext_tot
        all_eigvals = self._lc_all_eigvals
        all_eigvecs = self._lc_all_eigvecs
        density = self._lc_density
        foccs = self._lc_foccs

        # calculates the Kohn-Sham energy
        eks_density = self.eks_model(density) # energy density (nbatch, nr)
        Eks = self.grid.integralbox(eks_density, dim=-1) # (nbatch,)

        # calculate the individual non-interacting particles energy
        # sum_eigvals_list: list of (nbatch,)
        sum_eigvals_list = [(eigvals * focc).sum(dim=-1) \
            for (eigvals, focc) in zip(all_eigvals, foccs)]
        sum_eigvals = functools.reduce(lambda x,y: x+y, sum_eigvals_list)
        vks_integral = self.grid.integralbox(vks*density, dim=-1)

        # compute the interacting particles energy
        Etot = sum_eigvals - vks_integral + Eks
        self._lc_energy = Etot
        return Etot

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
    from ddft.hamiltons.hamiltonpw import HamiltonPlaneWave
    from ddft.hamiltons.hatomradial import HamiltonAtomRadial
    from ddft.modules.equilibrium import EquilibriumModule
    from ddft.grids.linearnd import LinearNDGrid
    from ddft.grids.radialgrid import LegendreRadialShiftExp
    from ddft.eks import BaseEKS, Hartree, xLDA

    class EKS1(BaseEKS):
        def __init__(self, a, p):
            super(EKS1, self).__init__()
            self.a = torch.nn.Parameter(a)
            self.p = torch.nn.Parameter(p)

        def forward(self, density):
            # small addition is made to safeguard if density equals to 0
            # this expression will be differentiated at least twice for 1st
            # order differentiation and more for higher order differentiation.
            vks = self.a * safepow(density.abs(), self.p)
            return vks

    dtype = torch.float64
    mode = "atom"
    if mode == "cartesian":
        ndim = 1
        boxshape = torch.tensor([31, 31, 31][:ndim])
        boxsizes = torch.tensor([10.0, 10.0, 10.0][:ndim], dtype=dtype)
        grid = LinearNDGrid(boxsizes, boxshape)
        H_model = HamiltonPlaneWave(grid)

        hparams = []

        rgrid = grid.rgrid
        rgrid_norm = (rgrid).norm(dim=-1)
        # vext = -1./(rgrid_norm + 1e-3)
        vext = rgrid_norm**2 * 0.5

    elif mode == "atom":
        gwidths = torch.logspace(np.log10(1e-5), np.log10(1e2), 100).to(dtype)
        grid = LegendreRadialShiftExp(1e-6, 1e4, 200, dtype=dtype)
        H_model = HamiltonAtomRadial(grid, gwidths, angmom=0)

        atomz = torch.tensor([1.0])
        hparams = [atomz]

        rgrid = grid.rgrid
        rgrid_norm = (rgrid).norm(dim=-1)
        vext = rgrid_norm*0

    H_model.to(dtype)
    vext = vext.unsqueeze(0).requires_grad_()

    nlowest = 2
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
    a = torch.tensor([0.1]).to(dtype)
    p = torch.tensor([1.3333]).to(dtype)
    focc = torch.tensor([[1.0, 0.0]]).requires_grad_() # (nbatch, nlowest)

    def getloss(a, p, vext, focc, return_model=False):
        # set up the modules
        eks_model = EKS1(a, p)
        eks_model = eks_model + Hartree()
        eks_model = eks_model + xLDA()
        eks_model.set_grid(H_model.grid)
        dft_model = DFT(H_model, eks_model, nlowest,
            **eigen_options)
        scf_model = EquilibriumModule(dft_model,
            forward_options=forward_options,
            backward_options=backward_options)

        # calculate the density
        nels = focc.sum(-1)
        density0 = torch.zeros_like(vext).to(vext.device) # _get_uniform_density(rgrid, nels)
        density = scf_model(density0, vext, focc, hparams)
        energy = dft_model.energy()

        # print(energy)
        # plt.plot(vext.view(-1).detach().numpy())
        # plt.plot(density.view(-1).detach().numpy())
        # plt.show()

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
    # vext_fd = finite_differences(getloss, (a, p, vext, focc), 2, eps=1e-3)
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

    # print("vext gradients:")
    # print(vext_grad)
    # print(vext_fd)
    # print(vext_grad / vext_fd)

    print("focc gradients:")
    print(focc_grad)
    print(focc_fd)
    print(focc_grad / focc_fd)
