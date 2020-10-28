from collections import namedtuple

__all__ = ["DensityInfo"]

# density info
_density_info_fields = [
    "density",  # torch.Tensor of density in the grid
    "gradn",  # tuple of torch.Tensor representing (gradx_n, grady_n, gradz_n)
    "laplacen",  # torch.Tensor of the laplace of the density
]
DensityInfo = namedtuple(
    "DensityInfo",
    _density_info_fields,
    defaults = (None,) * len(_density_info_fields))

def _add_densinfo(a, b):
    return DensityInfo(
        density = a.density + b.density,
        gradn = tuple((agn + bgn) for (agn, bgn) in zip(a.gradn, b.gradn)) \
                if a.gradn is not None else None,
        laplacen = a.laplacen + b.laplacen if a.laplacen is not None else None,
    )

def _mul_densinfo(a, f):
    assert not isinstance(f, DensityInfo)
    return DensityInfo(
        density = a.density * f,
        gradn = tuple((agn * f) for agn in a.gradn) if a.gradn is not None else None,
        laplacen = a.laplacen * f if a.laplacen is not None else None,
    )

DensityInfo.__add__ = _add_densinfo
DensityInfo.__mul__ = _mul_densinfo
