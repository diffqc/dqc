import torch
from ddft.modules.eigen import EigenModule
from ddft.modules.calcarith import DifferentialModule
from ddft.utils.misc import set_default_option

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

    Forward returns
    ---------------
    * new_density: torch.tensor (nbatch, nr)
        The new density calculated by one forward pass of the self-consistent
        field calculation.
    """
    def __init__(self, H_model, eks_model, nlowest, **eigen_options):
        super(DFT, self).__init__()
        eigen_options = set_default_option({
            "v_init": "randn",
        }, eigen_options)
        self.H_model = H_model
        self.eks_model = eks_model
        self.vks_model = DifferentialModule(eks_model)
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

        return new_density

    ############################# post processing #############################
    def energy(self):
        # calculate the total potential experienced by Kohn-Sham particles
        # from the last forward calculation
        vks = self._lc_vks
        vext_tot = self._lc_vext_tot
        eigvals = self._lc_eigvals
        eigvecs = self._lc_eigvecs
        density = self._lc_density
        focc = self._lc_focc

        # calculates the Kohn-Sham energy
        eks_density = self.eks_model(density) # energy density (nbatch, nr)
        Eks = self.H_model.integralbox(eks_density, dim=-1) # (nbatch,)

        # calculate the individual non-interacting particles energy
        sum_eigvals = (eigvals * focc).sum(dim=-1) # (nbatch,)
        vks_integral = self.H_model.integralbox(vks*density, dim=-1)

        # compute the interacting particles energy
        Etot = sum_eigvals - vks_integral + Eks
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
    from ddft.hamiltons.hspatial1d import HamiltonSpatial1D
    from ddft.hamiltons.hamiltonpw import HamiltonPlaneWave
    from ddft.modules.equilibrium import EquilibriumModule
    from ddft.spaces.qspace import QSpace

    class EKS1(torch.nn.Module):
        def __init__(self, a, p):
            super(EKS1, self).__init__()
            self.a = torch.nn.Parameter(a)
            self.p = torch.nn.Parameter(p)

        def forward(self, density):
            vks = self.a * density.abs()**self.p
            return vks

    dtype = torch.float64
    ndim = 3
    boxshape = [31, 31, 31][:ndim]
    boxsizes = [10.0, 10.0, 10.0][:ndim]
    rgrids = [torch.linspace(-boxsize/2., boxsize/2., nx+1)[:-1].to(dtype) for (boxsize,nx) in zip(boxsizes,boxshape)]
    rgrids = torch.meshgrid(*rgrids) # (nx,ny,nz)
    rgrid = torch.cat([rgrid.unsqueeze(-1) for rgrid in rgrids], dim=-1).view(-1,ndim) # (nr,3)
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
    a = torch.tensor([0.1]).to(dtype)
    p = torch.tensor([1.3333]).to(dtype)
    rgrid_norm = (rgrid).norm(dim=-1)
    # vext = -1./(rgrid_norm + 1e-3)
    vext = rgrid_norm**2 * 0.5
    vext = vext.unsqueeze(0).requires_grad_()
    focc = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).requires_grad_() # (nbatch, nlowest)

    def getloss(a, p, vext, focc, return_model=False):
        # set up the modules
        qspace = QSpace(rgrid, boxshape)
        H_model = HamiltonPlaneWave(qspace)
        eks_model = EKS1(a, p)
        dft_model = DFT(H_model, eks_model, nlowest,
            **eigen_options)
        scf_model = EquilibriumModule(dft_model,
            forward_options=forward_options,
            backward_options=backward_options)

        # calculate the density
        nels = focc.sum(-1)
        density0 = torch.zeros_like(vext).to(vext.device) # _get_uniform_density(rgrid, nels)
        density = scf_model(density0, vext, focc)
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
