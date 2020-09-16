from abc import abstractmethod
import xitorch as xt

class BaseQCCalc(xt.EditableModule):
    #################### postprocess functions ####################
    @abstractmethod
    def energy(self):
        """
        The total energy of the system in Hartree.
        """
        pass

    @abstractmethod
    def density(self, gridpts=None):
        """
        The electron density of the system in the specified grid points.
        If gridpts is None, then use the grid points from the system.
        """
        pass

    #################### editable module functions ####################
    @abstractmethod
    def getparamnames(self, methodname, prefix=""):
        pass
