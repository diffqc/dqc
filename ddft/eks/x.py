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

    def forward(self, density_up, density_dn, gradn_up=None, gradn_dn=None):
        # Ex(nu, nd) = 0.5 * (Ex(2*nu) + Ex(2*nd))

        if id(density_up) == id(density_dn):
            density = density_up + density_dn
            gradn = _sum_tuple(gradn_up, gradn_dn)
            return self._forward(density, gradn=gradn)
        else:
            eu = self._forward(
                2 * density_up,
                gradn=_sum_tuple(gradn_up, gradn_up),
            )
            ed = self._forward(
                2 * density_dn,
                gradn=_sum_tuple(gradn_dn, gradn_dn),
            )
            return 0.5 * (eu + ed)

    def potential(self, density_up, density_dn, gradn_up=None, gradn_dn=None):
        # Vx_u(nu, nd) = Vx(2*nu)
        # Vx_d(nu, nd) = Vx(2*nd)
        if id(density_up) == id(density_dn):
            density = density_up + density_dn
            gradn = _sum_tuple(gradn_up, gradn_dn)
            pot = self._potential(density, gradn=gradn)
            return pot, pot
        else:
            potu = self._potential(
                2 * density_up,
                gradn=_sum_tuple(gradn_up, gradn_up),
            )
            potd = self._potential(
                2 * density_dn,
                gradn=_sum_tuple(gradn_dn, gradn_dn),
            )
            return potu, potd

################## helper functions ##################

def _sum_tuple(tuple1, tuple2):
    if tuple1 is not None:
        res = tuple((tup1 + tup2) for (tup1, tup2) in zip(tuple1, tuple2))
    else:
        res = None
    return res
