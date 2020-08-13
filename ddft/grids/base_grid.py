from abc import abstractmethod, abstractproperty
import torch
import lintorch as lt

class BaseGrid(lt.EditableModule):

    @abstractmethod
    def get_dvolume(self):
        """
        Obtain the torch.tensor containing the dV elements for the integration.

        Returns
        -------
        * dV: torch.tensor (nr,)
            The dV elements for the integration
        """
        pass

    @abstractmethod
    def solve_poisson(self, f):
        """
        Solve Poisson's equation del^2 v = f, where f is a torch.tensor with
        shape (nbatch, nr) and v is also similar.
        The solve-Poisson's operator must be written in a way that it is
        a symmetric transformation when multiplied by get_dvolume().

        Arguments
        ---------
        * f: torch.tensor (nbatch, nr)
            Input tensor

        Results
        -------
        * v: torch.tensor (nbatch, nr)
            The tensor that fulfills del^2 v = f. Please note that v can be
            non-unique due to ill-conditioned operator del^2.
        """
        pass

    @abstractproperty
    def rgrid(self):
        """
        Returns (nr, ndim) torch.tensor which represents the spatial position
        of the grid.
        """
        pass

    @abstractproperty
    def boxshape(self):
        """
        Returns the shape of the signal in the real spatial dimension.
        prod(boxshape) == rgrid.shape[0]
        """
        pass

    ###################### integrals ######################
    def integralbox(self, p, dim=-1, keepdim=False):
        """
        Performing the integral over the spatial grid of the signal `p` where
        the signal in spatial grid is located at the dimension `dim`.

        Arguments
        ---------
        * p: torch.tensor (..., nr, ...)
            The tensor to be integrated over the spatial grid.
        * dim: int
            The dimension where it should be integrated.
        * keepdim: bool
            If True, then make the dimension into 1. Otherwise, omit the
            integration dimension.
        """
        if dim != -1:
            p = p.transpose(dim,-1)
        res = torch.matmul(p, self.get_dvolume().unsqueeze(-1))
        if dim != -1:
            res = res.transpose(dim,-1)
        if not keepdim:
            res = res.squeeze(dim)
        return res

    def mmintegralbox(self, p1, p2):
        """
        Perform the equivalent of matrix multiplication but replacing the sum
        with integral sum.

        Arguments
        ---------
        * p1: torch.tensor (..., n1, nr)
        * p2: torch.tensor (..., nr, n2)
            The integrands of the matrix multiplication integration.

        Returns
        -------
        * mm: torch.tensor (..., n1, n2)
            The result of matrix multiplication integration.
        """
        pleft = p1 * self.get_dvolume()
        return torch.matmul(pleft, p2)

    ###################### interpolate ######################
    def interpolate(self, f, rq, extrap=None):
        """
        Interpolate the function f to any point rq.

        Arguments
        ---------
        * f: torch.tensor (nbatch, nr)
            The function to be interpolated.
        * rq: torch.tensor (nrq, ndim)
            The position where the interpolated value is queried.
        * extrap: callable(torch.tensor) -> torch.tensor
            The extrapolation function. If None, it will be filled with 0.

        Returns
        -------
        * fq: torch.tensor (nbatch, nrq)
            The interpolated function at the given queried position.
        """
        raise RuntimeError("Unimplemented interpolate function for class %s" % \
                           (self.__class__.__name__))

    ###################### derivatives ######################
    def grad(self, p, idim, dim=-1):
        """
        Get the gradient in `idim` directions where `idim` indicate the index
        of the axis according to `self.rgrid`.

        Arguments
        ---------
        * p: torch.tensor (..., nr, ...)
            The function to be taken the derivative.
        * idim: int
            The index of the axis where the derivative is calculated
        * dim: int
            The dimension of the spatial information.

        Returns
        -------
        * dp: torch.tensor (..., nr, ...)
            The derivative in the `idim` axis.
        """
        raise RuntimeError("Grad for grid %s has not been implemented" % \
              self.__class__.__name__)

    def magnitude_derivative(self, p, dim=-1):
        """
        Returns the magnitude of the derivative |\nabla(p)|.

        Arguments
        ---------
        * p: torch.tensor (..., nr, ...)
            The function to be taken the derivative.
        * dim: int
            The dimension of the spatial information.

        Returns
        -------
        * dp: torch.tensor (..., nr, ...)
            The magnitude of the derivative of p.
        """
        if dim != -1:
            p = p.transpose(dim, -1)

        ndim = self.rgrid.shape[-1]
        pder_sq = 0
        for i in range(ndim):
            pder = self.grad(p, idim=i, dim=-1)
            pder_sq = pder_sq + pder*pder
        pder = torch.sqrt(pder_sq)

        # transpose back
        if dim != -1:
            pder = pder.transpose(dim, -1)
        return pder

    def laplace(self, p, dim=-1):
        """
        Returns the laplacian of a function p, i.e. \nabla^2(p).

        Arguments
        ---------
        * p: torch.tensor (..., nr, ...)
            The function to be taken the laplacian.
        * dim: int
            The dimension of the spatial information.

        Returns
        -------
        * dp: torch.tensor (..., nr, ...)
            The laplacian of p.
        """
        raise RuntimeError("Laplace for grid %s has not been implemented" % \
              self.__class__.__name__)

    ################### editable module ###################
    def getparamnames(self, methodname, prefix=""):
        if methodname == "integralbox" or methodname == "mmintegralbox":
            return self.getparamnames("get_dvolume", prefix=prefix)
        else:
            raise KeyError("getparamnames has no %s method" % methodname)


class Base3DGrid(BaseGrid):
    @abstractproperty
    def rgrid_in_xyz(self):
        """
        Returns the rgrid in Cartesian coordinate.
        """
        pass

    @abstractmethod
    def rgrid_to_xyz(self, rg):
        """
        Convert the coordinate rg (nrq, 3) to the cartesian coordinate (nrq, 3)
        """
        pass

    @abstractmethod
    def xyz_to_rgrid(self, xyz):
        """
        Convert the coordinate xyz (nrq, 3) to the rgrid coordinate (nrq, 3)
        """
        pass

class BaseRadialAngularGrid(Base3DGrid):
    @abstractproperty
    def radial_grid(self):
        """
        Returns the radial grid associated with the parent grid.
        """
        pass

class BaseMultiAtomsGrid(Base3DGrid):
    @abstractproperty
    def atom_grids(self):
        """
        Returns the grids for individual atom.
        """
        pass

    @abstractmethod
    def get_atom_weights(self):
        """
        Returns the weights associated with the integration grid per atom.
        Shape: (natoms, nr//natoms)
        """
        pass
