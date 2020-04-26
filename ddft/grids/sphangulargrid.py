import os
import warnings
import torch
import numpy as np
import ddft
from ddft.grids.base_grid import BaseGrid, BaseTransformed1DGrid, BaseRadialAngularGrid
from ddft.utils.spharmonics import spharmonics, vspharmonics

class Lebedev(BaseRadialAngularGrid):
    def __init__(self, radgrid, prec, basis_maxangmom=None, dtype=torch.float, device=torch.device('cpu')):
        super(Lebedev, self).__init__()

        # radgrid must be a BaseTransformed1DGrid
        if not isinstance(radgrid, BaseTransformed1DGrid):
            raise TypeError("Argument radgrid must be a BaseTransformed1DGrid")

        # cached variables
        self._basis_ = None
        self._deriv_basis_ = [None, None]
        self._angmoms_ = None

        # the precision must be an odd number in range [3, 131]
        self.prec = prec
        self.basis_maxangmom = basis_maxangmom if basis_maxangmom is not None else prec
        assert (prec % 2 == 1) and (3 <= prec <= 131),\
               "Precision must be an odd number between 3 and 131"

        self.dtype = dtype
        self.device = device

        # load the datasets from https://people.sc.fsu.edu/~jburkardt/datasets/sphere_lebedev_rule/sphere_lebedev_rule.html
        dset_path = os.path.join(os.path.split(ddft.__file__)[0], "datasets", "lebedevquad", "lebedev_%03d.txt"%prec)
        assert os.path.exists(dset_path), "The dataset lebedev_%03d.txt does not exist" % prec
        lebedev_dsets = torch.tensor(np.loadtxt(dset_path), dtype=dtype, device=device)
        self.phithetargrid = lebedev_dsets[:,:2] / 180.0 * np.pi # (nphitheta,2)
        self.wphitheta = lebedev_dsets[:,-1] # (nphitheta)
        nphitheta = self.phithetargrid.shape[0]

        # get the radial grid
        self.radgrid = radgrid
        self.radrgrid = radgrid.rgrid[:,0] # (nrad,)
        nrad = self.radrgrid.shape[0]
        self.nrad = nrad

        # combine the grids
        # (nrad, nphitheta)
        rg = self.radrgrid.unsqueeze(-1).repeat(1, nphitheta).unsqueeze(-1) # (nrad, nphitheta, 1)
        ptg = self.phithetargrid.unsqueeze(0).repeat(nrad, 1, 1) # (nrad, nphitheta, 2)
        self._rgrid = torch.cat((rg, ptg), dim=-1).view(-1, 3) # (nrad*nphitheta, 3)
        self._rgrid_xyz = None

        # get the integration part
        self._dvolume_rad = self.radgrid.get_dvolume() # (nrad,)
        # print(self._dvolume_rad.sum()/self.radrgrid.max()**3/(4*np.pi/3), self.wphitheta.sum())
        dvolume = self._dvolume_rad.unsqueeze(-1) * self.wphitheta
        self._dvolume = dvolume.view(-1) # (nrad*nphitheta)

        # # check the basis orthonormality
        # basis = self._get_basis() # (nsh, nphitheta)
        # basis_olp = torch.matmul(basis*self.wphitheta, basis.transpose(-2,-1))
        # assert torch.allclose(basis_olp, torch.eye(basis_olp.shape[0], dtype=basis_olp.dtype, device=basis_olp.device))
        # raise RuntimeError

    def get_dvolume(self):
        return self._dvolume

    def solve_poisson(self, f):
        # f: (nbatch, nr)
        # nr = nrad * nphitheta
        # the expression below is used to satisfy the following conditions:
        # * symmetric operator (by doing the integral 1/|r-r1|)
        # * 0 at r=\infinity, but not 0 at the bound (again, by doing the integral 1/|r-r1|)
        # to satisfy all the above, we choose to do the integral of
        #     Vlm(r) = integral_rmin^rmax (rless^l) / (rgreat^(l+1)) flm(r1) r1^2 dr1
        # where rless = min(r,r1) and rgreat = max(r,r1)

        # get the spherical harmonics components of f as function of radius
        eps = 1e-12
        nbatch, nr = f.shape
        f1 = f.view(nbatch, self.nrad, -1) # (nbatch, nrad, nphitheta)
        basis = self._get_basis() # (nsh, nphitheta)
        basis_integrate = self._get_basis(basis_integrate=True)
        frad_lm = torch.bmm(basis_integrate.unsqueeze(0).expand(nbatch,-1,-1), f1.transpose(-2,-1)) # (nbatch, nsh, nrad)

        # the computation is done by computing the ratio of rless/rgreat first
        # then integrate it
        # it is done this way to prevent numerical instability, although the
        # computation is slightly more expensive

        # calculate the matrix rless / rgreat
        angmoms = self._get_angmoms().unsqueeze(-1) # (nsh,1)
        angmoms1 = angmoms.unsqueeze(-1)
        rless = torch.min(self.radrgrid.unsqueeze(-1), self.radrgrid) # (nrad, nrad)
        rgreat = torch.max(self.radrgrid.unsqueeze(-1), self.radrgrid)
        rratio = (rless / rgreat)**angmoms1 * 1.0/rgreat # (nsh, nrad, nrad)

        # the integralbox for radial grid is integral[4*pi*r^2 f(r) dr] while here
        # we only need to do integral[f(r) dr]. That's why it is divided by (4*np.pi)
        # and it is not multiplied with (self.radrgrid**2) in the lines below
        intgn = (frad_lm).unsqueeze(-2) * rratio # (nbatch, nsh, nrad, nrad)
        vrad_lm = self.radgrid.integralbox(intgn, dim=-1) / ((2*angmoms+1) * (4*np.pi))

        # convert back to the spatial basis
        v = torch.matmul(vrad_lm.transpose(-2,-1), basis) # (nbatch, nrad, nphitheta)
        v = v.view(nbatch, nr)
        return -v

    def interpolate(self, f, rq, extrap=None):
        # f: (nbatch, nr)
        # rq: (nr2, ndim)
        # return (nbatch, nr2)

        # obtain the basis part of f
        nbatch = f.shape[0]
        nr2 = rq.shape[0]
        f1 = f.view(nbatch, self.nrad, -1) # (nbatch, nrad, nphitheta)
        basis_integrate = self._get_basis(basis_integrate=True)
        frad_lm = torch.bmm(basis_integrate.unsqueeze(0).expand(nbatch,-1,-1), f1.transpose(-2,-1)) # (nbatch, nsh, nrad)

        # obtain the points to be interpolated and extrapolated
        rqrad = rq[:,0] # (nr2)
        rmax = self.radrgrid.max()
        idxinterp = rqrad <= rmax
        idxextrap = rqrad > rmax
        allinterp = torch.all(idxinterp)
        if allinterp:
            rqradinterp = rq[:,:1]
            phithetaqinterp = rq[:,1:]
        else:
            rqradinterp = rq[idxinterp,:1]
            phithetaqinterp = rq[idxinterp,1:]

        # interpolate f in r-direction
        frqrad = self.radgrid.interpolate(frad_lm.view(-1, frad_lm.shape[-1]), rqradinterp).view(nbatch, -1, rqradinterp.shape[0]) # (nbatch, nsh, nr2interp)
        # get the basis Y as function of rq
        rqbasis = self._get_basis(phithetaqinterp) # (nsh, nr2interp)
        # get the value by multiplying and sum the radial function and the basis
        frqinterp = (frqrad * rqbasis).sum(dim=1) # (nbatch, nr2interp)

        # if there is no extrapolation, then we're done here
        if allinterp:
            return frqinterp

        # extrapolate the function
        if extrap is not None:
            frqextrap = extrap(rq[idxextrap,:]) # (nbatch, nr2extrap)

        # combine the interpolation and extrapolation
        frq = torch.zeros((nbatch, nr2), dtype=rq.dtype, device=rq.device)
        frq[:,idxinterp] = frqinterp
        if extrap is not None:
            frq[:,idxextrap] = frqextrap

        return frq

    def grad(self, p, idim, dim=-1):
        if dim != -1:
            p = p.transpose(dim, -1) # (..., nr)

        batch_size = p.shape[:-1]
        p = p.view(*batch_size, self.nrad, -1) # (..., nrad, nphitheta)
        if idim == 0:
            # radial derivative
            pres = self.radgrid.grad(p, idim=0, dim=-2).reshape(*batch_size, -1) # (..., nr)

        else:
            # phi (azimuth) or theta derivative
            basis_integrate = self._get_basis(basis_integrate=True) # (nsh, nphitheta)
            deriv_basis = self._get_deriv_basis(idim-1) # (nsh, nphitheta)
            psh = torch.matmul(p, basis_integrate.transpose(-2,-1)) # (..., nrad, nsh)
            pres = torch.matmul(psh, deriv_basis) / (self.radgrid.rgrid[:,:1] + 1e-15) # (..., nrad, nphitheta)

            pres = pres.view(*batch_size, -1) # (..., nr)

        if dim != -1:
            pres = pres.transpose(dim, -1)
        return pres

    def laplace(self, p, dim=-1):
        if dim != -1:
            p = p.transpose(dim, -1)

        batch_size = p.shape[:-1]
        p = p.view(*batch_size, self.nrad, -1) # (..., nrad, nphitheta)
        basis = self._get_basis() # (nsh, nphitheta)
        basis_integrate = self._get_basis(basis_integrate=True) # (nsh, nphitheta)

        # get the spherical harmonics components
        rs = self.radgrid.rgrid[:,:1] # (nrad, 1)
        psh = torch.matmul(p, basis_integrate.transpose(-2,-1)) # (..., nrad, nsh)
        angmoms = self._get_angmoms() # (nsh,)
        pang = -angmoms * (angmoms+1) * psh / (rs*rs + 1e-12) # (..., nrad, nsh)
        pphitheta = torch.matmul(pang, basis) # (..., nrad, nphitheta)

        # get the contribution from the radial direction
        prad = self.radgrid.laplace(p, dim=-2) # (..., nrad, nphitheta)

        # add all contributions
        res = (prad + pphitheta).reshape(*batch_size, -1)

        if dim != -1:
            res = res.transpose(dim, -1)
        return res

    @property
    def radial_grid(self):
        return self.radgrid

    @property
    def rgrid(self):
        return self._rgrid

    @property
    def rgrid_in_xyz(self):
        if self._rgrid_xyz is None:
            self._rgrid_xyz = self.rgrid_to_xyz(self._rgrid)
        return self._rgrid_xyz

    def rgrid_to_xyz(self, rg):
        r = rg[:,0]
        phi = rg[:,1]
        theta = rg[:,2]

        rsintheta = r * torch.sin(theta)
        x = (rsintheta * torch.cos(phi)).unsqueeze(-1)
        y = (rsintheta * torch.sin(phi)).unsqueeze(-1)
        z = (r*torch.cos(theta)).unsqueeze(-1)
        xyz = torch.cat((x,y,z), dim=-1)
        return xyz

    def xyz_to_rgrid(self, xyz):
        x = xyz[:,0]
        y = xyz[:,1]
        z = xyz[:,2]
        xy = torch.sqrt(x*x + y*y)
        r = torch.sqrt(x*x + y*y + z*z).unsqueeze(-1)
        theta = torch.atan2(xy, z).unsqueeze(-1)
        phi = torch.atan2(y, x).unsqueeze(-1)
        return torch.cat((r, phi, theta), dim=-1)

    @property
    def boxshape(self):
        warnings.warn("Boxshape is obsolete. Please refrain in using it.")

    def _get_basis(self, phitheta=None, basis_integrate=False):
        if phitheta is None:
            if self._basis_ is None:
                phi = self.phithetargrid[:,0]
                costheta = torch.cos(self.phithetargrid[:,1])
                self._basis_ = spharmonics(costheta, phi, self.basis_maxangmom)
                self._basis_integrate_ = self._basis_ * self.wphitheta
            if basis_integrate:
                return self._basis_integrate_
            else:
                return self._basis_
        else:
            phi = phitheta[:,0]
            costheta = torch.cos(phitheta[:,1])
            basis = spharmonics(costheta, phi, self.basis_maxangmom)
            if basis_integrate:
                return basis * self.wphitheta
            else:
                return basis

    def _get_deriv_basis(self, iphitheta, phitheta=None):
        if phitheta is None:
            if self._deriv_basis_[iphitheta] is None:
                phi = self.phithetargrid[:,0]
                costheta = torch.cos(self.phithetargrid[:,1])
                self._deriv_basis_[iphitheta] = vspharmonics(iphitheta, costheta, phi, self.basis_maxangmom)
            return self._deriv_basis_[iphitheta]
        else:
            phi = phitheta[:,0]
            costheta = torch.cos(phitheta[:,1])
            return vspharmonics(iphitheta, costheta, phi, self.basis_maxangmom)

    def _get_angmoms(self):
        if self._angmoms_ is None:
            lhat = []
            for angmom in range(self.basis_maxangmom+1):
                lhat = lhat + [angmom]*(2*angmom+1)
            self._angmoms_ = torch.tensor(lhat, dtype=self.dtype, device=self.device) # (nsh,)
        return self._angmoms_

    #################### editable module parts ####################
    def getparams(self, methodname):
        if methodname == "solve_poisson":
            return [self._basis_, self._basis_integrate_, self.radrgrid] + \
                    self.radgrid.getparams("get_dvolume")
        elif methodname == "interpolate":
            return [self._basis_integrate_] + self.radgrid.getparams("interpolate")
        elif methodname == "get_dvolume":
            return [self._dvolume]
        else:
            raise RuntimeError("The method %s has not been specified for getparams" % methodname)

    def setparams(self, methodname, *params):
        if methodname == "solve_poisson":
            self._basis_, self._basis_integrate_, self.radrgrid = params[:3]
            return 3 + self.radgrid.setparams("get_dvolume", *params[3:])
        elif methodname == "interpolate":
            self._basis_integrate_ = params[0]
            return 1 + self.radgrid.setparams("interpolate", *params[1:])
        elif methodname == "get_dvolume":
            self._dvolume, = params[:1]
            return 1
        else:
            raise RuntimeError("The method %s has not been specified for setparams" % methodname)

if __name__ == "__main__":
    import lintorch as lt
    from ddft.grids.radialgrid import LegendreRadialShiftExp
    radgrid = LegendreRadialShiftExp(1e-4, 1e2, 100, dtype=torch.float64)
    grid = Lebedev(radgrid, prec=5, basis_maxangmom=4, dtype=torch.float64)
    rgrid = grid.rgrid.clone().detach()
    f = torch.exp(-rgrid[:,0].unsqueeze(0)**2*0.5)

    lt.list_operating_params(grid.solve_poisson, f)
    lt.list_operating_params(grid.interpolate, f, rgrid)
    lt.list_operating_params(grid.get_dvolume)
