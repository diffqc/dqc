def orthogonalize(a, b, dim=-1):
    """
    Orthogonalize a with respect to b.
    The orthogonalization is done such that (a*b).sum(dim=dim) is equal to 0
    """
    proj = (a * b).sum(dim=dim, keepdim=True) / (b * b).sum(dim=dim, keepdim=True)
    return a - proj * b
