import torch
import xitorch as xt
from typing import List
import dqc.hamilton.intor as intor
from dqc.df.base_df import BaseDF
from dqc.utils.datastruct import DensityFitInfo

class DFMol(BaseDF):
    """
    DFMol represents the class of density fitting for an isolated molecule.
    """
    def __init__(self, dfinfo: DensityFitInfo, wrapper: intor.LibcintWrapper):
        self.dfinfo = dfinfo
        self.wrapper = wrapper
        self._is_built = False

    def build(self) -> BaseDF:
        self._is_built = True

        # construct the matrix used to calculate the electron repulsion for
        # density fitting method
        method = self.dfinfo.method
        auxbasiswrapper = intor.LibcintWrapper(self.dfinfo.auxbases,
                                               spherical=self.wrapper.spherical)
        basisw, auxbw = intor.LibcintWrapper.concatenate(self.wrapper, auxbasiswrapper)

        if method == "coulomb":
            j2c = intor.coul2c(auxbw)  # (nxao, nxao)
            j3c = intor.coul3c(basisw, other1=basisw,
                               other2=auxbw)  # (nao, nao, nxao)
        elif method == "overlap":
            j2c = intor.overlap(auxbw)  # (nxao, nxao)
            # TODO: implement overlap3c
            raise NotImplementedError(
                "Density fitting with overlap minimization is not implemented")
        self._j2c = j2c  # (nxao, nxao)
        self._j3c = j3c  # (nao, nao, nxao)
        self._inv_j2c = torch.inverse(j2c)

        self._el_mat = torch.matmul(j3c, self._inv_j2c)  # (nao, nao, nxao)
        return self

    def get_elrep(self, dm: torch.Tensor) -> xt.LinearOperator:
        # dm: (*BD, nao, nao)
        # elrep_mat: (nao, nao, nao, nao)
        # return: (*BD, nao, nao)

        df_coeffs = torch.einsum("...ij,ijk->...k", dm, self._el_mat)  # (*BD, nxao)
        mat = torch.einsum("...k,ijk->...ij", df_coeffs, self._j3c)  # (*BD, nao, nao)
        mat = (mat + mat.transpose(-2, -1)) * 0.5
        return xt.LinearOperator.m(mat, is_hermitian=True)

    @property
    def j2c(self) -> torch.Tensor:
        return self._j2c

    @property
    def j3c(self) -> torch.Tensor:
        return self._j3c

    def getparamnames(self, methodname: str, prefix: str = "") -> List[str]:
        if methodname == "get_elrep":
            return [prefix + "_el_mat", prefix + "_j3c"]
        else:
            raise KeyError("getparamnames has no %s method" % methodname)
