import torch

def legint(coeffs, dim=-1, zeroat="left"):
    # Integrate the coefficients of Legendre polynomial one time
    # The resulting integral always have coeffs[0] == 0.
    # coeffs: (..., nr)

    c = coeffs.transpose(dim, -1)
    n = c.shape[-1]
    j21 = 1 + 2 * torch.arange(n).to(coeffs.dtype).to(coeffs.device)

    if zeroat == "left":
        m0 = 1.0
    elif zeroat == "right":
        m0 = -1.0
    else:
        m0 = 0.0

    dmid = c[..., :-2] / j21[:-2] - c[..., 2:] / j21[2:]
    dr = c[..., -2:] / j21[-2:]
    dl = -c[..., 1:2] / 3.0 + m0 * c[..., :1]

    res = torch.cat((dl, dmid, dr), dim=-1).transpose(dim, -1)
    return res

def legval(x, order):
    if order == 0:
        return x*0 + 1
    elif order == 1:
        return x
    elif order == 2:
        return 1.5*x**2 - 0.5
    elif order == 3:
        return 2.5*x**3 - 1.5*x
    elif order == 4:
        return 4.375*x**4 - 3.75*x**2 + 0.375
    elif order == 5:
        return 7.875*x**5 - 8.75*x**3 + 1.875*x
    else:
        raise RuntimeError("The legendre polynomial order %d has not been implemented" % order)

def assoclegval(cost, l, m):
    sint = torch.sqrt(1-cost*cost)
    if l == 0:
        return cost*0 + 1
    elif l == 1:
        if m == 0:
            return cost
        elif m == 1:
            return -sint
    elif l == 2:
        if m == 0:
            return 1.5 * cost*cost - 0.5
        elif m == 1:
            return -3*cost*sint
        elif m == 2:
            return 3*sint*sint
    elif l == 3:
        if m == 0:
            return 2.5*cost**3 - 1.5*cost
        elif m == 1:
            return (-7.5*cost**2 + 1.5) * sint
        elif m == 2:
            return 15 * cost * sint*sint
        elif m == 3:
            return -15 * sint**3
    elif l == 4:
        if m == 0:
            return (35*cost**4 - 30*cost**2 + 3) / 8.0
        elif m == 1:
            return -2.5*(7*cost**3 - 3*cost) * sint
        elif m == 2:
            return 7.5*(7*cost**2 - 1)*sint**2
        elif m == 3:
            return -105*cost*sint**3
        elif m == 4:
            return 105*sint**4
    else:
        raise RuntimeError("The associated legendre polynomial order %d has not been implemented" % l)

if __name__ == "__main__":
    coeffs = torch.tensor([1., 2., 3.])
    print(legint(coeffs)) # (dc, 0.4, 0.6667, 0.6000)
