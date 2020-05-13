import numpy as np
import torch
from ddft.utils.legendre import legint, legder, assoclegval, assoclegval_iter

def test_legint_legder():
    coeffs = torch.tensor([1., 2., 3.])
    assert torch.allclose(legint(coeffs)[1:], torch.tensor([0.4, 2.0/3, 0.6])) # (dc, 0.4, 0.6667, 0.6000)
    assert torch.allclose(legder(coeffs), torch.tensor([2., 9., 0.])) # (2, 9, 0)
    coeffs2 = torch.tensor([1., 2., 3., 4., 5., 6.])
    assert torch.allclose(legder(coeffs2), torch.tensor([12, 24, 50, 35, 54, 0.])) # (12, 24, 50, 35, 54, 0)
    coeffs3 = torch.tensor([1., 2., 3., 4., 5., 6., 7.])
    assert torch.allclose(legder(coeffs3), torch.tensor([12, 45, 50, 84, 54, 77, 0.])) # (12, 45, 50, 84, 54, 77, 0)

def test_assoclegval_iter():
    cost = torch.linspace(-1., 1., 100)
    for l in range(8):
        for m in range(l+1):
            val1 = assoclegval(cost, l, m)
            val2 = assoclegval_iter(cost, l, m)
            print(l, m, (val1-val2).abs().max())
            assert torch.allclose(val1, val2, atol=1e-5)
