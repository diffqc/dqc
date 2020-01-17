import torch
import numpy as np

class HartreePlaneWave(torch.nn.Module):
    def __init__(self, space):
        super(HartreePlaneWave, self).__init__()
        self.space = space

        # get the 1/|q2|
        qgrid = self.space.qgrid # (ns,ndim)
        q2 = (qgrid*qgrid).sum(-1) # (ns,)
        inv_q2 = 1.0 / q2
        # zeroing the central grid because it cancels out with the ion potential
        inv_q2[q2.abs() < 1e-7] = 0.0
        self.inv_q2 = inv_q2.unsqueeze(-1).expand(1,2) # (ns, 2)

    def forward(self, density):
        # density: (nbatch, nr)

        # transform to Fourier domain
        density_ft = self.space.transformsig(density, dim=-1) # (nbatch, ns, 2)

        # multiply with 2*pi / |q|^2
        hartree_ft = density_ft * 2 * np.pi * self.inv_q2 # (nbatch, ns, 2)

        # transform it back
        Vhartree = self.space.invtransformsig(hartree_ft, dim=-2) # (nbatch, nr)
        Ehartree = Vhartree * density # the Hartree energy density

        return Ehartree
