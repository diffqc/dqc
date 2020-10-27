import numpy as np

def get_x_s2_alpha(density, gradn=None, tau=None, eps=1e-16):
    # convert the density, gradient, and the kinetic energy (tau)
    # into uniform exchange, s^2, and alpha
    n = density
    dn = gradn

    # uniform exchange calculation
    exunif2 = (3 * np.pi * np.pi * (n + eps)) ** (1. / 3)
    exunif2n = exunif2 * n
    exunif = -(0.75 / np.pi) * exunif2

    s2 = None
    alpha = None
    if dn is not None:
        # s calculation
        dn2 = sum(d * d for d in dn)
        s2 = dn2 / (4 * exunif2n * exunif2n + eps)

        if tau is not None:
            # alpha calculation
            tw = dn2 / (8 * n + eps)
            tunif = 0.3 * exunif2 * exunif2n
            alpha = (tau - tw) / (tunif + eps)

    return exunif, s2, alpha

def get_c_rs_zeta_t2_alpha(
        density_up, density_dn,
        gradn_up=None, gradn_dn=None,
        tau_up=None, tau_dn=None,
        eps=1e-16):
    # convert the density, gradient, and the kinetic energy (tau)
    # into uniform exchange, rs, zeta, t2, and alpha
    # See: Sun, et al., "Strongly Constrained and Appropriately Normed Semilocal
    #                    Density Functional"

    n_up = density_up
    n_dn = density_dn
    dn_up = gradn_up
    dn_dn = gradn_dn

    n = n_up + n_dn

    exunif2 = (3 * np.pi * np.pi * n + eps) ** (1. / 3)
    exunif2n = exunif2 * n
    exunif = -(0.75/np.pi) * exunif2
    rs = (np.pi / 0.75 * n + eps) ** (-1. / 3)
    zeta = (n_up - n_dn) / (n + eps)

    t2 = None
    alpha = None
    if dn_up is not None:
        # t2 calculation
        dn2 = sum((du + dd) ** 2 for (du, dd) in zip(dn_up, dn_dn))
        phi = 0.5 * (((1 + zeta) + eps) ** (2./3) + ((1 - zeta) + eps) ** (2./3))
        t2 = dn2 / (16*phi*phi * (3/np.pi) ** (1./3) * n ** (7./3) + eps)

        if tau_up is not None:
            # alpha calculation
            d = 0.5 * ((1 + zeta) ** (5./3) + (1 - zeta) ** (5./3))
            tw = dn2 / (8 * n + eps)
            tunif = 0.3 * exunif2 * exunif2 * n * d
            tau = tau_up + tau_dn
            alpha = (tau - tw) / (tunif + eps)

    return exunif, rs, zeta, t2, alpha
