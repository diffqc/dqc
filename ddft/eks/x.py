from abc import abstractmethod
from ddft.eks.base_eks import BaseEKS

__all__ = ["Exchange"]

class Exchange(BaseEKS):
    @abstractmethod
    def _forward(self, density, gradn):
        pass

    @abstractmethod
    def _potential(self, density, gradn):
        pass

    def forward(self, densinfo_u, densinfo_d):
        # Ex(nu, nd) = 0.5 * (Ex(2*nu) + Ex(2*nd))

        if id(densinfo_u) == id(densinfo_d):
            densinfo = densinfo_u + densinfo_d
            return self._forward(densinfo)
        else:
            eu = self._forward(densinfo_u + densinfo_u)
            ed = self._forward(densinfo_d + densinfo_d)
            return 0.5 * (eu + ed)

    def potential(self, densinfo_u, densinfo_d):
        # Vx_u(nu, nd) = Vx(2*nu)
        # Vx_d(nu, nd) = Vx(2*nd)
        if id(densinfo_u) == id(densinfo_d):
            pot = self._potential(densinfo_u + densinfo_d)
            return pot, pot
        else:
            potu = self._potential(densinfo_u + densinfo_u)
            potd = self._potential(densinfo_d + densinfo_d)
            return potu, potd
