import pytest
import torch
import warnings
from ddft.eks import xLDA, cLDA_PW
from ddft.utils.datastruct import DensityInfo

@pytest.mark.parametrize(
    "xccls",
    [xLDA, cLDA_PW],
)
def test_lda(xccls):
    density = torch.rand(10, dtype=torch.double)
    densinfo = DensityInfo(density = density)
    xc = xccls()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        xc.assertparams(xc.forward, densinfo_u=densinfo, densinfo_d=densinfo)
