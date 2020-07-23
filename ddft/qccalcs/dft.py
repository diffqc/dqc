import torch
from ddft.qccalcs.base_qccalc import BaseQCCalc
from ddft.modules.eigen import EigenModule
from ddft.modules.equilibrium import EquilibriumModule
from ddft.eks import VKS, Hartree, xLDA

__all__ = ["dft"]

class dft(BaseQCCalc):
    """
    Perform the Kohn-Sham DFT.
    """
    def __init__(self, system, eks_model="lda", vext_fcn=None,
            # arguments for scf run
            density0=None, eigen_options={}, fwd_options={}, bck_options={}):

        # extract properties of the system
        self.system = system
        self.focc = self.system.get_occupation().unsqueeze(0) # (nbatch=1,norb)
        self.grid = self.system._get_grid()
        self.hmodel = self.system._get_hamiltonian()
        nlowest = self.focc.shape[-1]
        self.dtype = self.system.dtype
        self.device = self.system.device

        # setup the external potential
        self.vext = self.__get_vext(vext_fcn)

        # set up the vhks
        eks_model = self.__get_eks_model(eks_model)
        self.eks_model = eks_model + Hartree()
        self.vks_model = VKS(self.eks_model, self.grid)

        # set up the eigen module for the forward pass and scf module
        eigen_options = self.__setup_eigen_options(eigen_options)
        self.eigen_model = EigenModule(self.hmodel, nlowest,
            rlinmodule=self.hmodel.overlap, **eigen_options)
        self.scf_model = EquilibriumModule(self.__forward_pass,
            forward_options=fwd_options, backward_options=bck_options)

        # set up the initial density before running scf module
        density0 = self.__get_init_density(density0)

        # run the self-consistent iterations
        self.scf_density = self.scf_model(density0)

        # postprocess properties
        self.scf_energy = None

    ######################### postprocess functions #########################
    def energy(self):
        if self.scf_energy is None:
            # calculate the total potential experienced by Kohn-Sham particles
            # from the last forward calculation
            density = self.scf_density
            eigvals, eigvecs, vks = self.__diagonalize(density)

            # calculates the Kohn-Sham energy
            eks_density = self.eks_model(density) # energy density (nbatch, nr)
            Eks = self.grid.integralbox(eks_density, dim=-1) # (nbatch,)

            # calculate the individual non-interacting particles energy
            sum_eigvals = (eigvals * self.focc).sum(dim=-1) # (nbatch,)
            vks_integral = self.grid.integralbox(vks*density, dim=-1)

            # compute the interacting particles energy
            Etot = sum_eigvals - vks_integral + Eks + self.system.get_nuclei_energy()
            self.scf_energy = Etot
        return self.scf_energy

    def density(self, gridpts=None):
        if gridpts is None:
            return self.scf_density

    def __forward_pass(self, density):
        # density: (nbatch, nr)
        eigvals, eigvecs, vks = self.__diagonalize(density)

        # normalize the norm of density
        eigvec_dens = self.hmodel.getdens(eigvecs) # (nbatch, nr, nlowest)
        dens = eigvec_dens * self.focc.unsqueeze(1) # (nbatch, nr, nlowest)
        new_density = dens.sum(dim=-1) # (nbatch, nr)

        return new_density

    def __diagonalize(self, density):
        # calculate the total potential experienced by Kohn-Sham particles
        vks = self.vks_model(density) # (nbatch, nr)
        vext_tot = self.vext + vks

        # compute the eigenpairs
        # evals: (nbatch, norb), evecs: (nbatch, nr, norb)
        hparams = (vext_tot,)
        rparams = []
        eigvals, eigvecs = self.eigen_model(hparams, rparams=rparams)

        return eigvals, eigvecs, vks

    ############# parameters setup functions #############
    def __get_eks_model(self, eks_model):
        if eks_model == "lda":
            return xLDA()
        else:
            raise RuntimeError("Unknown eks model: %s" % eks_model)

    def __get_init_density(self, density0):
        # if there is no specified density0, then use one-time forward pass as the initial guess
        if density0 is None:
            density0 = torch.zeros_like(self.grid.rgrid[:,0]).unsqueeze(0).to(self.device) # (nbatch, nr)
            with torch.no_grad():
                density0 = self.__forward_pass(density0)

        if isinstance(density0, torch.Tensor) and len(density0.shape) == 1:
            density0 = density0.unsqueeze(0) # (nbatch, nr)

        density0 = density0.detach()
        return density0

    def __get_vext(self, vext_fcn):
        if vext_fcn is not None:
            vext = vext_fcn(self.grid.rgrid).unsqueeze(0) # (nbatch=1, nr)
        else:
            vext = torch.zeros_like(self.grid.rgrid[:,0]).unsqueeze(0) # (nbatch, nr)
        return vext

    def __setup_eigen_options(self, options):
        nsize = self.hmodel.shape[0]
        if nsize < 100:
            options["method"] = "exacteig"
        return options

    ############################# editable module #############################
    def getparams(self, methodname):
        if methodname == "__forward_pass":
            return [self.focc] + self.hmodel.getparams("getdens") + self.getparams("__diagonalize")
        elif methodname == "__diagonalize":
            return [self.vext] + self.eigen_model.getparams("__call__") + self.vks_model.getparams("__call__")
        else:
            raise RuntimeError("The method %s is not defined for getparams"%methodname)

    def setparams(self, methodname, *params):
        if methodname == "__forward_pass":
            self.focc, = params[:1]
            idx = 1
            idx += self.hmodel.setparams("getdens", *params[idx:])
            idx += self.setparams("__diagonalize", *params[idx:])
            return idx
        elif methodname == "__diagonalize":
            self.vext, = params[:1]
            idx = 1
            idx += self.eigen_model.setparams("__call__", *params[idx:])
            idx += self.vks_model.setparams("__call__", *params[idx:])
            return idx
        else:
            raise RuntimeError("The method %s is not defined for setparams"%methodname)

if __name__ == "__main__":
    from ddft.systems.mol import mol
    m = mol("Be 0 0 0", basis="6-311++G**")
    scf = dft(m)
    print(scf.energy())
