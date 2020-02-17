import torch

__all__ = ["VKS"]

class VKS(torch.nn.Module):
    def __init__(self, eks_model):
        super(VKS, self).__init__()
        self.eks_model = eks_model

    def forward(self, x):
        assert x.ndim == 2, "The input to VKS module must be 2-dimensional tensor (nbatch, nrgrid)"
        if x.requires_grad:
            xinp = x
        else:
            xinp = x.clone().requires_grad_()

        with torch.enable_grad():
            y = self.eks_model(xinp) # (nbatch,nr)
            ysum = y.sum()
        grad_enabled = torch.is_grad_enabled()
        dx = torch.autograd.grad(ysum, (xinp,),
            create_graph=grad_enabled)[0]
        return dx # (same shape as x)
