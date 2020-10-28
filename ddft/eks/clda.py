import torch
from ddft.eks.base_eks import BaseEKS
from ddft.eks.utils import get_c_rs_zeta_t2_alpha

__all__ = ["cLDA_PW"]

class cLDA_PW(BaseEKS):
    def __init__(self):
        self.a_pp     = torch.tensor([[1.], [1.], [1.]])
        self.a_a      = torch.tensor([[0.0310907], [0.01554535], [0.0168869]])
        self.a_alpha1 = torch.tensor([[0.21370], [ 0.20548], [ 0.11125]])
        self.a_beta1  = torch.tensor([[7.5957], [14.1189], [10.357]])
        self.a_beta2  = torch.tensor([[3.5876], [6.1977], [3.6231]])
        self.a_beta3  = torch.tensor([[1.6382], [3.3662], [ 0.88026]])
        self.a_beta4  = torch.tensor([[0.49294], [0.62517], [0.49671]])
        self.a_fz20   = torch.tensor(1.709920934161365617563962776245)

    def forward(self, densinfo_u, densinfo_d):
        density_up = densinfo_u.density
        density_dn = densinfo_d.density

        exunif, rs, zeta, t2, alpha = get_c_rs_zeta_t2_alpha(
            density_up, density_dn
        )

        g_aux = self.a_beta1 * torch.sqrt(rs) + \
                self.a_beta2 * rs + self.a_beta3 * rs ** 1.5 + \
                self.a_beta4 * rs ** (self.a_pp + 1)
        g     = -2 * self.a_a * (1 + self.a_alpha1 * rs) * \
                torch.log1p(1. / (2 * self.a_a * g_aux))

        f_zeta = ((1 + zeta) ** (4./3) + (1 - zeta)**(4./3) - 2) / (2 ** (4./3) - 2)
        f_pw = g[0] + zeta**4 * f_zeta*(g[1] - g[0] + g[2] / self.a_fz20) - \
               f_zeta * g[2] / self.a_fz20
        return f_pw * (density_up + density_dn)

    def getfwdparamnames(self, prefix=""):
        return [
            prefix + "a_pp",
            prefix + "a_a",
            prefix + "a_alpha1",
            prefix + "a_beta1",
            prefix + "a_beta2",
            prefix + "a_beta3",
            prefix + "a_beta4",
            prefix + "a_fz20",
        ]
