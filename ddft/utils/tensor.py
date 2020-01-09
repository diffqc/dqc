import torch

def roll_1(x, n):
    """
    Roll to the right of the first dimension (zero-based).
    """
    return torch.cat((x[:, -n:], x[:, :-n]), dim=1)
