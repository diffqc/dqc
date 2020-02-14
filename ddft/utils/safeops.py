import torch

def safepow(a, p, eps=1e-9):
    return _safepow.apply(a, p, eps)

class _safepow(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, p, eps):
        ctx.a = a
        ctx.p = p
        ctx.res = a**p
        ctx.eps = eps
        return ctx.res

    @staticmethod
    def backward(ctx, grad_res):
        a = ctx.a.clone()
        a[a.abs() < ctx.eps] = ctx.eps
        gr = grad_res * ctx.res
        grad_a = None
        grad_p = None
        if isinstance(ctx.a, torch.Tensor):
            grad_a = gr * ctx.p / a
        if isinstance(ctx.p, torch.Tensor):
            grad_p = gr * torch.log(a)
        return (grad_a, grad_p, None)
