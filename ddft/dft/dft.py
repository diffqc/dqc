import functools
import torch
import numpy as np
import lintorch as lt
from ddft.modules.eigen import EigenModule
from ddft.utils.misc import set_default_option, unpack
from ddft.utils.safeops import safepow
from ddft.eks import VKS

class DFT(lt.EditableModule):
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

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, density, vext, focc, *params):
        # density: (nbatch, nr)
        # vext: (nbatch, nr)
        # focc: (nbatch, nlowest)

        eigvals, eigvecs, vks = self._diagonalize(density, vext, focc, *params)

        # normalize the norm of density
        eigvec_dens = self.H_model.getdens(eigvecs) # (nbatch, nr, nlowest)
        dens = eigvec_dens * focc.unsqueeze(1) # (nbatch, nr, nlowest)
        new_density = dens.sum(dim=-1) # (nbatch, nr)

        return new_density

    ############################# post processing #############################
    def energy(self, density, vext, focc, *params):
        # calculate the total potential experienced by Kohn-Sham particles
        # from the last forward calculation
        eigvals, eigvecs, vks = self._diagonalize(density, vext, focc, *params)

        # calculates the Kohn-Sham energy
        eks_density = self.eks_model(density) # energy density (nbatch, nr)
        Eks = self.H_model.grid.integralbox(eks_density, dim=-1) # (nbatch,)

        # calculate the individual non-interacting particles energy
        sum_eigvals = (eigvals * focc).sum(dim=-1) # (nbatch,)
        vks_integral = self.H_model.grid.integralbox(vks*density, dim=-1)

        # compute the interacting particles energy
        Etot = sum_eigvals - vks_integral + Eks
        return Etot

    ############################# helper functions #############################
    def _diagonalize(self, density, vext, focc, *params):
        # unpack the parameters
        hparams, rparams = unpack(params, [self.H_model.nhparams, self.H_model.nolp_params])

        # calculate the total potential experienced by Kohn-Sham particles
        vks = self.vks_model(density) # (nbatch, nr)
        vext_tot = vext + vks

        # compute the eigenpairs
        # evals: (nbatch, nlowest), evecs: (nbatch, nr, nlowest)
        eigvals, eigvecs = self.eigen_model((vext_tot, *hparams), rparams=rparams)

        return eigvals, eigvecs, vks

    ############################# editable module #############################
    def getparams(self, methodname):
        if methodname == "forward" or methodname == "__call__":
            return self.vks_model.getparams("__call__") + \
                   self.eigen_model.getparams("__call__") + \
                   self.H_model.getparams("getdens")
        else:
            raise RuntimeError("The method %s is not defined for getparams"%methodname)

    def setparams(self, methodname, *params):
        if methodname == "forward" or methodname == "__call__":
            idx = self.vks_model.setparams("__call__", *params)
            idx += self.eigen_model.setparams("__call__", *params[idx:])
            idx += self.H_model.setparams("getdens", *params[idx:])
            return idx
        else:
            raise RuntimeError("The method %s is not defined for setparams"%methodname)

class DFTMulti(lt.EditableModule):
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

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, density, vext, foccs, *params):
        # density: (nbatch, nr)
        # vext: (nbatch, nr)
        # foccs: list of (nbatch, nlowest)

        all_eigvals, all_eigvecs, vks = self._diagonalize(density, vext, foccs, *params)

        # normalize the norm of density
        # dens: list of (nbatch, nr)
        dens = [(H_model.getdens(eigvecs) * focc.unsqueeze(1)).sum(dim=-1) \
            for (H_model, eigvecs, focc)\
            in zip(self.H_models, all_eigvecs, foccs)] # (nbatch, nr, nlowest)
        new_density = functools.reduce(lambda x,y: x + y, dens)

        return new_density

    ############################# post processing #############################
    def energy(self, density, vext, foccs, *params):
        # calculate the total potential experienced by Kohn-Sham particles

        all_eigvals, all_eigvecs, vks = self._diagonalize(density, vext, foccs, *params)

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

    ############################# helper functions #############################
    def _diagonalize(self, density, vext, foccs, *params):
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
        return all_eigvals, all_eigvecs, vks

    ############################# editable module #############################
    def getparams(self, methodname):
        if methodname == "forward" or methodname == "__call__":
            res = self.vks_model.getparams("__call__")
            for eigen_model in self.eigen_models:
                res = res + eigen_model.getparams("__call__")
            for H_model in self.H_models:
                res = res + H_model.getparams("getdens")
            return res
        else:
            raise RuntimeError("The method %s is not defined for getparams"%methodname)

    def setparams(self, methodname, *params):
        if methodname == "forward" or methodname == "__call__":
            idx = self.vks_model.setparams("__call__", *params)

            for eigen_model in self.eigen_models:
                idx += eigen_model.setparams("__call__", *params[idx:])

            for H_model in self.H_models:
                idx += H_model.setparams("getdens", *params[idx:])
            return idx
        else:
            raise RuntimeError("The method %s is not defined for setparams"%methodname)


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
