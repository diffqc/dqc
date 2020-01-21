import torch
from ddft.modules.base_linear import BaseLinearModule

class RealModule(BaseLinearModule):
    """
    Taking only the real part of complex linear module.
    This is useful in getting the eigenvalues and eigenvectors to remove the
    degeneracy due to the complex number representation.
    The complex number dimension is at the end of the vector
    representation.
    For signal with shape (...,nr,...) for real value, the complex
    representation will be (...,nr,2,...).

    __init__ arguments:
    * model: BaseLinearModule instance
        The complex linear model that has shape of (ns,ns) where ns must be
        even.
    """
    def __init__(self, model):
        super(RealModule, self).__init__()

        # check model
        _check_model(model)
        self.model = model
        self._shape = [model.shape[0]//2, model.shape[1]//2]

    def forward(self, x, *params):
        # x: (nbatch, ns//2) or (nbatch, ns//2, nelmts)

        # add the imaginary part as zero
        x = add_zero_imag(x, dim=1)
        y = self.model(x, *params) # (nbatch, ns, nelms)
        yreal = get_real_part(y, dim=1)
        return yreal

    @property
    def shape(self):
        return self._shape

    def diag(self, *params):
        modeldiag = self.model.diag(*params) # (nbatch, ns)
        modeldiag = modeldiag.view(modeldiag.shape[0], -1, 2) # (nbatch,ns//2,2)
        return modeldiag[:,:,0] # only take the real part

def add_zero_imag(x, dim=1):
    d = dim+1
    xzero = torch.zeros_like(x).to(x.device)
    y = torch.cat((x.unsqueeze(d), xzero.unsqueeze(d)), dim=d) # (nbatch, ns//2, 2)
    y = y.view(*y.shape[:dim], -1, *y.shape[dim+2:])
    return y

def get_real_part(y, dim=1):
    # y: (...,ns,...)
    y = y.view(*y.shape[:dim], -1, 2, *y.shape[dim+1:]) # (...,ns//2,2,...)
    yreal = y.index_select(dim+1, torch.LongTensor([0])).squeeze(dim+1)
    return yreal

def _check_model(model):
    if not isinstance(model, BaseLinearModule):
        raise TypeError("The model must be an instance of BaseLinearModule")
    if model.shape[0] % 2 != 0 or model.shape[1] % 2 != 0:
        raise ValueError("The dimension of the model must be even")
    if len(model.shape) != 2:
        raise ValueError("The model must be a 2 dimensional matrix")
