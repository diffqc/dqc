import torch
from ddft.utils.misc import set_default_option
from ddft.maths.rootfinder import lbfgs

class EquilibriumModule(torch.nn.Module):
    """
    Equilibrium module evaluates the output of a model such that

        y = model(y, *params)

    The model should take input (y,*params) and produce output of y.
    """
    def __init__(self, model, forward_options={}, backward_options={}):
        super(EquilibriumModule, self).__init__()
        self.model = model
        self.fwd_options = forward_options
        self.bck_options = backward_options

    def forward(self, y0, *params):
        # y0 & each of params: (nbatch, ...)
        yequi = _Forward.apply(self.model, y0, self.fwd_options, params)
        if self.training:
            yequi.requires_grad_()
            ymodel = self.model(yequi, *params)
            yequi = _Backward.apply(ymodel, yequi, self.bck_options)
        return yequi

class _Forward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, model, y0, options, orig_params):
        # set default options
        config = set_default_option({
            "max_niter": 50,
            "min_reps": 1e-6,
            "verbose": False,
        }, options)

        def loss(y):
            ymodel = model(y, *orig_params)
            return y - ymodel

        # solve the equilibrium equation with L-BFGS
        jinv0 = 1.0
        y = lbfgs(loss, y0, jinv0=jinv0, **config)
        return y

    @staticmethod
    def backward(ctx, *grads):
        res = tuple([None for _ in range(4)])
        return res

class _Backward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ymodel, yinp, options):
        ctx.options = set_default_option({
            "max_niter": 50,
            "min_reps": 1e-6,
            "verbose": False,
        }, options)

        ctx.ymodel = ymodel
        ctx.yinp = yinp
        return ymodel

    @staticmethod
    def backward(ctx, grad_yout):
        # grad_ymodel: (nbatch,...)
        ymodel = ctx.ymodel
        yinp = ctx.yinp

        def _apply_ImDfDy(gy):
            dfdy, = torch.autograd.grad(ymodel, (yinp,), gy, retain_graph=True)
            return gy - dfdy

        # solve x for  (I-df/dy)(x) == gy
        def loss_for_inv(x):
            return _apply_ImDfDy(x) - grad_yout

        gx0 = torch.zeros_like(grad_yout).to(grad_yout.device)
        jinv0 = 1.0
        gymodel = lbfgs(loss_for_inv, gx0, jinv0=jinv0, **ctx.options)

        return (gymodel, None, None)

if __name__ == "__main__":
    import time
    from ddft.utils.fd import finite_differences

    class DummyModule(torch.nn.Module):
        def __init__(self, A):
            super(DummyModule, self).__init__()
            self.A = torch.nn.Parameter(A)

        def forward(self, y, x):
            # y: (nbatch, nr)
            # x: (nbatch, nr)
            nbatch = y.shape[0]
            tanh = torch.nn.Tanh()
            A = self.A.unsqueeze(0).expand(nbatch, -1, -1)
            Ay = torch.bmm(A, y.unsqueeze(-1)).squeeze(-1)
            Ayx = Ay + x
            return tanh(0.1 * Ayx)

    dtype = torch.float64
    nr = 7
    nbatch = 1
    A  = torch.randn((nr, nr)).to(dtype)
    x  = torch.rand((nbatch, nr)).to(dtype).requires_grad_()
    y0 = torch.rand((nbatch, nr)).to(dtype).requires_grad_()

    model = DummyModule(A)
    eqmodel = EquilibriumModule(model)
    y = eqmodel(y0, x)

    print("Forward results:")
    print(y)
    print(model(y, x))
    print("    should be close to 1:")
    print(y / model(y, x))

    def getloss(A, x, y0, return_model=False):
        model = DummyModule(A)
        eqmodel = EquilibriumModule(model)
        y = eqmodel(y0, x)
        loss = (y*y).sum()
        if not return_model:
            return loss
        else:
            return loss, model

    # gradient with backprop
    loss, model = getloss(A, x, y0, return_model=True)
    loss.backward()
    A_grad = list(model.parameters())[0].grad.data
    x_grad = x.grad.data

    # gradient with finite_differences
    with torch.no_grad():
        A_fd = finite_differences(getloss, (A, x, y0), 0, eps=1e-5)
        x_fd = finite_differences(getloss, (A, x, y0), 1, eps=1e-5)

    print("Gradient of A:")
    print(A_grad)
    print(A_fd)
    print("    should be close to 1:")
    print(A_grad / A_fd)

    print("Gradient of x:")
    print(x_grad)
    print(x_fd)
    print("    should be close to 1:")
    print(x_grad / x_fd)
