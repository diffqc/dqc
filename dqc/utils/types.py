import torch

def get_complex_dtype(dtype: torch.dtype) -> torch.dtype:
    # return the corresponding complex type given the real floating point datatype
    if dtype == torch.float64:
        return torch.complex128
    elif dtype == torch.float32:
        return torch.complex64
    else:
        raise TypeError("Unsupported datatype %s for conversion to complex" % dtype)
