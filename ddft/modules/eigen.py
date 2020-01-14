import torch
from ddft.modules.base_linear import BaseLinearModule
from ddft.maths.eigpairs import davidson

class EigenModule(torch.nn.Module):
    """
    Module to wrap a linear module to obtain `nlowest` eigenpairs.

    __init__ arguments:
    -------------------
    * linmodule: BaseLinearModule
        The linear module whose forward signature is `forward(self, x, *params)`
        and it should provide the gradient to `x` and each of `params`.
        The linmodule must be a square matrix with size (na, na)
    * nlowest: int
        Indicates how many lowest eigenpairs should be retrieved by this module.

    forward arguments:
    ------------------
    * *params: list of differentiable torch.tensor
        The parameters to be passed to linmodule forward pass.
        The shape of each params should be (nbatch, ...)

    forward returns:
    ----------------
    * eigvals: (nbatch, nlowest)
    * eigvecs: (nbatch, na, nlowest)
        The eigenvalues and eigenvectors of linear transformation module.
    """
    def __init__(self, linmodule, nlowest, **options):
        super(Eigendecompose, self).__init__()

        self.linmodule = linmodule
        self.nlowest = nlowest
        self.options = options

        # check type
        if not isinstance(self.linmodule, BaseLinearModule):
            raise TypeError("The linmodule argument must be instance of BaseLinearModule")

    def forward(self, *params):
        eigvals, eigvecs = davidson(self.linmodule, self.nlowest,
            params, **self.options)
        return eigvals, eigvecs
