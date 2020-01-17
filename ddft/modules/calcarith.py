from functools import reduce
import torch

class DifferentialModule(torch.nn.Module):
    """
    Differential module calculates d(sum(output))/d(input) where `input` is
    a batched tensor of any size.

    In DFT context, the differential module can be used to obtain the Kohn-Sham
    potential from a model that calculates the Kohn-Sham energy.
    """
    def __init__(self, model):
        super(DifferentialModule, self).__init__()
        self.model = model

    def forward(self, x):
        if x.requires_grad:
            xinp = x
        else:
            xinp = x.clone().requires_grad_()

        with torch.enable_grad():
            y = self.model(xinp) # (nbatch,1^n)
            ysum = y.sum()
        grad_enabled = torch.is_grad_enabled()
        dx = torch.autograd.grad(ysum, (xinp,),
            create_graph=grad_enabled)[0]
        return dx # (same shape as x)

class AddModule(torch.nn.Module):
    def __init__(self, *models):
        super(AddModule, self).__init__()
        self.models = models

    def forward(self, x):
        xs = [m(x) for m in self.models]
        return reduce(lambda x,y: x+y, xs)

class NegModule(torch.nn.Module):
    def __init__(self, model):
        super(NegModule, self).__init__()
        self.model = model

    def forward(self, x):
        return -self.model(x)
