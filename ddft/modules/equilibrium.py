import torch
import lintorch as lt
from ddft.utils.misc import set_default_option
from ddft.maths.rootfinder import lbfgs, selfconsistent, broyden

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
            "min_eps": 1e-6,
            "verbose": False,
            "linesearch": True,
            "jinv0": 0.5,
            "method": "lbfgs",
        }, options)

        def loss(y):
            ymodel = model(y, *orig_params)
            # return ((y - ymodel)**2).sum()
            return y - ymodel

        # solve the equilibrium equation with L-BFGS
        # y = approximate_newton(loss, y0, **config)

        # yvar = y0.detach().requires_grad_()
        # opt = torch.optim.LBFGS([yvar], line_search_fn="strong_wolfe")
        # for i in range(config["max_niter"]):
        #     def closure():
        #         opt.zero_grad()
        #         with torch.enable_grad():
        #             lss = loss(yvar)
        #         lss.backward()
        #
        #         if config["verbose"]:
        #             print("Iter %3d: loss %.3e" % (i+1, lss.data))
        #         return lss
        #     opt.step(closure)
        # y = yvar.data

        jinv0 = config["jinv0"]
        method = config["method"].lower()
        if method == "lbfgs":
            y = lbfgs(loss, y0, **config)
        elif method == "selfconsistent":
            y = selfconsistent(loss, y0, **config)
        elif method == "broyden":
            y = broyden(loss, y0, **config)
        else:
            raise RuntimeError("Unknown method: %s" % config["method"])
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
            "min_eps": 1e-9,
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

        nr = ymodel.shape[-1]
        @lt.module(shape=(nr,nr), is_symmetric=False)
        def _apply_ImDfDy(gy, ymodel, yinp):
            gy = gy.squeeze(-1)
            dfdy, = torch.autograd.grad(ymodel, (yinp,), gy,
                retain_graph=True, create_graph=torch.is_grad_enabled())
            res = gy - dfdy
            res = res.unsqueeze(-1)
            return res

        gymodel = lt.solve(_apply_ImDfDy, [ymodel, yinp], grad_yout.unsqueeze(-1),
            fwd_options=ctx.options, bck_options=ctx.options).squeeze(-1)
        return (gymodel, None, None)

def approximate_newton(lossfn, y0, **config):
    # y0: (nbatch, nr)
    # y: (nbatch, nr, 1)
    y = y0.unsqueeze(-1)
    verbose = config["verbose"]
    min_eps = config["min_eps"]

    nr = y0.shape[-1]
    for i in range(config["max_niter"]):
        yinp = y.clone().detach().requires_grad_()
        with torch.enable_grad():
            loss = lossfn(yinp.squeeze(-1)).unsqueeze(-1) # (nbatch, nr, 1)

        @lt.module(shape=(nr,nr), is_symmetric=False)
        def jac(y, loss, yinp):
            return torch.autograd.grad((loss,), (yinp,), grad_outputs=(y,),
                retain_graph=True, create_graph=torch.is_grad_enabled())[0]

        # check convergence
        maxloss = loss.abs().max()
        if maxloss < min_eps:
            break
        if verbose:
            print("Iter %3d: maxloss %.3e" % (i+1, maxloss))

        dy = lt.solve(jac, [loss, yinp], loss, fwd_options={"min_eps": 1e-2,
            "method": "conjgrad",
            "verbose": True}) # (nbatch, nr, 1)
        y = y - dy
        del loss
        del yinp

    return y.squeeze(-1)

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
