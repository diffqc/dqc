import torch
from ddft.modules.eigen import EigenModule
from ddft.utils.misc import set_default_option

class DFT1D(torch.nn.Module):
    def __init__(self, H_model, vks_model, nlowest, **eigen_options):
        super(DFT1D, self).__init__()
        eigen_options = set_default_option({
            "v_init": "randn",
        }, eigen_options)
        self.vks_model = vks_model
        self.eigen_model = EigenModule(H_model, nlowest, **eigen_options)
        self.sum_wf2 = H_model.sumwf2()

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
        eigvec_dens = (eigvecs*eigvecs) * self.sum_wf2 # (nbatch, nr, nlowest)
        dens = eigvec_dens * focc.unsqueeze(1) # (nbatch, nr, nlowest)
        new_density = dens.sum(dim=-1) # (nbatch, nr)

        return new_density
