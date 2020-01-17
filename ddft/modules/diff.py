import torch

class DifferentialModule(torch.nn.Module):
    """
    Differential module calculates d(output)/d(input) where `output` is a value
    per batch and input is a batched tensor of any size.

    In DFT context, the differential module can be used to obtain the Kohn-Sham
    potential from a model that calculates the Kohn-Sham energy.
    """
    def __init__(self, model):
        super(DifferentialModule, self).__init__()
        self.model = model

    def forward(self, x):
        y = self.model(x) # (nbatch,1^n)
        ysum = y.sum()
        dx = torch.autograd.grad(ysum, (x,),
            retain_graph=True, create_graph=True)
        return dx # (same shape as x)
