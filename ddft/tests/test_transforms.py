import torch
from ddft.transforms.base_transform import *
from ddft.transforms.tensor_transform import *

def test_identity_1():
    # test with a single vector in x

    A = IdentityTransform((2, 5, 5), 3)
    B = IdentityTransform((2, 5, 5), 1)
    C = A + B
    D = A - B
    E = A * B
    x = torch.tensor([[1.0, 2.1, 1.4, 0.2, -0.1], [0.4, -0.2, 0.4, 0.6, 1.9]])

    assert torch.allclose(C(x), x*4)
    assert torch.allclose(C.T(x), x*4)
    assert torch.allclose(C.inv()(x), x/4.0, rtol=1e-6, atol=1e-8)

    assert torch.allclose(D(x), x*2)
    assert torch.allclose(D.T(x), x*2)
    assert torch.allclose(D.inv()(x), x/2.0, rtol=1e-6, atol=1e-8)

    assert torch.allclose(E(x), x*3)
    assert torch.allclose(E.T(x), x*3)
    assert torch.allclose(E.inv()(x), x/3.0, rtol=1e-6, atol=1e-8)

def test_identity_2():
    # test with multiple vectors in x

    A = IdentityTransform((2, 5, 5), 3)
    B = IdentityTransform((2, 5, 5), 1)
    C = A + B
    D = A - B
    E = A * B

    # x is (2, 5, 2)
    x = torch.tensor(
       [[[0.9831, 0.4194],
         [0.8660, 0.2629],
         [0.7393, 0.8659],
         [0.7117, 0.3268],
         [0.3056, 0.5452]],

        [[0.8737, 0.2454],
         [0.6001, 0.1051],
         [0.2872, 0.1382],
         [0.2007, 0.8182],
         [0.2370, 0.2582]]])

    assert torch.allclose(C(x), x*4)
    assert torch.allclose(C.T(x), x*4)
    assert torch.allclose(C.inv()(x), x/4.0, rtol=1e-6, atol=1e-8)

    assert torch.allclose(D(x), x*2)
    assert torch.allclose(D.T(x), x*2)
    assert torch.allclose(D.inv()(x), x/2.0, rtol=1e-6, atol=1e-8)

    assert torch.allclose(E(x), x*3)
    assert torch.allclose(E.T(x), x*3)
    assert torch.allclose(E.inv()(x), x/3.0, rtol=1e-6, atol=1e-8)
