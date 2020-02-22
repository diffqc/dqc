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

if __name__ == "__main__":
    coeffs = torch.tensor([1., 2., 3.])
    print(legint(coeffs)) # (dc, 0.4, 0.6667, 0.6000)
