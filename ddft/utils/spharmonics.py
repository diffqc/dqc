import torch
import numpy as np
from ddft.utils.legendre import legval, deriv_assoclegval, deriv_assoclegval_azimuth

def spharmonics(cost, phi, maxangmom):
    # ref: https://www.overleaf.com/read/kwbhpjfdzyvt
    nsh = (maxangmom+1)**2
    nphitheta = phi.shape[0]
    sint = torch.sqrt(1-cost*cost)

    dtype = cost.dtype
    device = cost.device
    angbasis = torch.empty((nsh, nphitheta), dtype=dtype, device=device)
    angmoms = torch.empty((nsh,), dtype=dtype, device=device)
    yml = torch.ones_like(cost).to(device)
    yml_m1 = torch.ones_like(cost).to(device)
    row = 0
    for m in range(maxangmom+1):
        yml_orig = yml
        nrm = np.sqrt(2.) if m != 0 else 1.0
        for l in range(m, maxangmom+1):
            angbasis[row] = yml * nrm * torch.cos(m*phi)
            angmoms[row] = l
            row += 1
            if m != 0:
                angbasis[row] = yml * nrm * torch.sin(m*phi)
                angmoms[row] = l
                row += 1
            if l == maxangmom:
                continue

            # prepare the value for the next l-iteration
            yml_m2 = yml_m1
            yml_m1 = yml
            if l == m:
                yml = cost * yml_m1 * np.sqrt(2*l+3.)
            else:
                a = np.sqrt((2*l+3.) / (l-m+1.) / (l+m+1.))
                yml = (cost * yml_m1 * (np.sqrt(2*l+1.) * a) - yml_m2 * (a*np.sqrt((l*l-m*m) / (2*l-1.))) )

        if m == maxangmom:
            continue

        # prepare for the next m-iteration
        yml_m1 = yml_orig
        yml = np.sqrt((2*m+3.)/(2*m+2.)) * sint * yml_m1
    return angbasis, angmoms

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
                alegcostheta = deriv_assoclegval_azimuth(costheta, l, m) # already divided by sin(theta)
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
