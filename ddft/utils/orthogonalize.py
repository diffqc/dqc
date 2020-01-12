def orthogonalize(self, a, b):
    """
    Orthogonalize a with respect to b.
    Shape of a & b: (nbatch, nfeat)
    """
    proj = (a * b).sum(dim=-1) / (b * b).sum(dim=-1)
    return a - proj.unsqueeze(-1) * b
