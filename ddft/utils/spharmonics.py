import torch
import numpy as np
from ddft.utils.legendre import legval, assoclegval, deriv_assoclegval

def spharmonics(costheta, phi, maxangmom, onlytheta=False):
    # costheta: (nphitheta,)
    # phi: (nphitheta,)
    # maxangmom: int
    # return: (nsh, phitheta)

    nsh = (maxangmom+1)**2 if not onlytheta else (maxangmom+1)
    nphitheta = phi.shape[0]

    dtype = phi.dtype
    device = phi.device
    angbasis = torch.empty((nsh, nphitheta), dtype=dtype, device=device)
    angbasis_row = 0
    for l in range(maxangmom+1):
        legcostheta = legval(costheta, l) # (nphitheta,)

        # m = 0
        normm0 = np.sqrt(2*l+1)
        angbasis[angbasis_row] = legcostheta * normm0
        angbasis_row += 1

        # m != 0
        if not onlytheta:
            nm = 0.5
            for m in range(1,l+1):
                alegcostheta = assoclegval(costheta, l, m)
                nm = nm * (l-m+1) * (l+m)
                norm = normm0 / np.sqrt(nm)
                angbasis[angbasis_row] = alegcostheta * torch.cos(m*phi) * norm
                angbasis_row += 1
                angbasis[angbasis_row] = alegcostheta * torch.sin(m*phi) * norm
                angbasis_row += 1
    return angbasis

def vspharmonics(iphitheta, costheta, phi, maxangmom):
    # costheta: (nphitheta,)
    # phi: (nphitheta,)
    # maxangmom: int
    # return: (nsh, phitheta)
    nsh = (maxangmom+1)**2
    nphitheta = phi.shape[0]
    dtype = phi.dtype
    device = phi.device

    angbasis = torch.zeros((nsh, nphitheta), dtype=dtype, device=device)
    angbasis_row = 0
    if iphitheta == 0: # derivative in phi
        for l in range(maxangmom+1):
            legcostheta = legval(costheta, l) # (nphitheta,)

            # m = 0
            normm0 = np.sqrt(2*l+1)
            # angbasis[angbasis_row] = 0.0
            angbasis_row += 1

            # m != 0
            nm = 0.5
            for m in range(1,l+1):
                alegcostheta = assoclegval(costheta, l, m)
                nm = nm * (l-m+1) * (l+m)
                norm = normm0 / np.sqrt(nm)
                angbasis[angbasis_row] = -alegcostheta * torch.sin(m*phi) * m * norm
                angbasis_row += 1
                angbasis[angbasis_row] =  alegcostheta * torch.cos(m*phi) * m * norm
                angbasis_row += 1

    elif iphitheta == 1:
        for l in range(maxangmom+1):
            legcostheta = deriv_assoclegval(costheta, l, 0) # (nphitheta,)

            # m = 0
            normm0 = np.sqrt(2*l+1)
            angbasis[angbasis_row] = legcostheta * normm0
            angbasis_row += 1

            # m != 0
            nm = 0.5
            for m in range(1,l+1):
                alegcostheta = deriv_assoclegval(costheta, l, m)
                nm = nm * (l-m+1) * (l+m)
                norm = normm0 / np.sqrt(nm)
                angbasis[angbasis_row] = alegcostheta * torch.cos(m*phi) * norm
                angbasis_row += 1
                angbasis[angbasis_row] = alegcostheta * torch.sin(m*phi) * norm
                angbasis_row += 1
    else:
        raise RuntimeError("Input to iphitheta can only be 0 (for phi) or 1 (for theta).")
    return angbasis
