import torch
from ddft.transforms.base_transform import BaseTransform

"""
NTransform is the operator for (dv_KS / dn)^T.
"""

class NTransform(BaseTransform):
    def __init__(self, vks_model, density):
        self.vks_model = vks_model

        # build the graph from density to vks
        self.density_temp = density.detach().clone().to(density.device).requires_grad_()
        with torch.enable_grad():
            self.vks_temp = vks_model(self.density_temp) # (nbatch, nr)

        # obtain the shape and dtype
        nbatch, nr = self.density_temp.shape
        self._shape = (nbatch, nr, nr)

    def _forward(self, x):
        # x will be (nbatch, nr) and it is grad_vks
        # the output is (nbatch, nr)
        res, = torch.autograd.grad(self.vks_temp, (self.density_temp,),
            grad_outputs=x,
            retain_graph=True)
        return res

    def _transpose(self, y):
        raise RuntimeError("Unimplemented transpose method for %s" % (self.__class__.__name__))

    @property
    def shape(self):
        return self._shape

    @property
    def dtype(self):
        return self.density_temp.dtype
