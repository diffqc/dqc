import pytest
import torch
import warnings
from ddft.eks import xLDA, cLDA_PW

@pytest.mark.parametrize(
    "xccls",
    [xLDA, cLDA_PW],
)
def test_lda(xccls):
    density = torch.rand(10, dtype=torch.double)
    xc = xccls()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        xc.assertparams(xc.forward, density_up=density, density_dn=density)
