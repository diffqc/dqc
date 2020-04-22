import torch
from ddft.grids.base_grid import BaseGrid

class LinearNDGrid(BaseGrid):
    def __init__(self, boxsizes, boxshape):
        self.boxsizes = boxsizes
        self._boxshape = boxshape

        assert boxsizes.shape[0] == boxshape.shape[0], \
               "The boxsizes and boxshape arguments must have the same length"

        self.device = self.boxsizes.device
        self.dtype = self.boxsizes.dtype

        # set up the grid
        ndim = boxsizes.shape[0]
        self.ndim = ndim
        rgrids = [torch.linspace(-boxsize/2., boxsize/2., nx+1)[:-1].to(self.dtype).to(self.device)\
                  for (boxsize,nx) in zip(boxsizes,boxshape)]
        rgrids = torch.meshgrid(*rgrids) # (nx,ny,nz)
        self._rgrid = torch.cat([rgrid.unsqueeze(-1) for rgrid in rgrids], dim=-1).view(-1,self.ndim) # (nr,3)

        # get the pixel size
        idx = 0
        allshape = (*boxshape, self._rgrid.shape[-1])
        m = 1
        for i in range(self.ndim,0,-1):
            m *= allshape[i]
            idx += m
        pixsize = self._rgrid[idx,:] - self._rgrid[0,:] # (ndim,)
        self.dr3 = torch.prod(pixsize).unsqueeze(0).repeat(torch.prod(boxshape))

    def get_dvolume(self):
        return self.dr3

    def solve_poisson(self, f):
        raise RuntimeError("Unimplemented solve_poisson for LinearNDGrid")

    @property
    def rgrid(self):
        return self._rgrid

    @property
    def boxshape(self):
        return self._boxshape

    #################### editable module parts ####################
    def getparams(self, methodname):
        if methodname == "get_dvolume":
            return [self.dr3]
        else:
            raise RuntimeError("The method %s has not been specified for getparams" % methodname)

    def setparams(self, methodname, *params):
        if methodname == "get_dvolume":
            self.dr3, = params
        else:
            raise RuntimeError("The method %s has not been specified for setparams" % methodname)
