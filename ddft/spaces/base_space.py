from abc import abstractmethod, abstractproperty

class BaseSpace(object):
    @abstractproperty
    def K(self):
        pass

    @abstractmethod
    def H(self):
        pass
