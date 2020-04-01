import torch
from scipy.special import gammainc, gamma

def incgamma(s, x):
    return _incgamma_fcn.apply(s, x)

class _incgamma_fcn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, s, x):
        # s: int
        # x: tensor
        # return: same shape as x
        ctx.x = x
        ctx.s = s
        xnp = x.detach().numpy()
        res = gammainc(s, xnp) * gamma(s)
        return torch.tensor(res, dtype=x.dtype, device=x.device)

    @staticmethod
    def backward(ctx, grad_output):
        s = ctx.s
        x = ctx.x
        grad_x = x**(s-1) * torch.exp(-x) * grad_output
        return (None, grad_x)
