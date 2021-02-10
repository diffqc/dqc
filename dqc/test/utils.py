import gc
import torch
from typing import Callable

__all__ = ["assert_no_memleak_tensor"]

# memory test functions
def assert_no_memleak_tensor(fcn: Callable, strict: bool = True, gccollect: bool = False):
    """
    Assert no tensor memory leak when calling the function.

    Arguments
    ---------
    fcn: Callable
        A function with no input and output to be checked.
    strict: bool
        If True, then there must be no additional tensor allocated after it
        exits the function.
    gccollect: bool
        If True, then manually apply ``gc.collect()`` after the function
        execution.

    Exceptions
    ----------
    AssertionError
        Raised if there is an indication of memory leak in the function.
    """
    size0 = _get_tensor_memory()
    ntries = 10
    if strict:
        fcn()
        if gccollect:
            gc.collect()
        size = _get_tensor_memory()
        if size0 != size:
            _show_memsize(fcn, ntries, gccollect=gccollect)
        assert size0 == size
    else:
        raise NotImplementedError("Option non-strict memory leak checking has not been implemented")

def _show_memsize(fcn, ntries: int = 10, gccollect: bool = False):
    # show the memory growth
    size0 = _get_tensor_memory()
    for i in range(ntries):
        fcn()
        if gccollect:
            gc.collect()
        size = _get_tensor_memory()
        print("%3d iteration: %.7f MiB of tensors" % (i + 1, size - size0))

def _get_tensor_memory() -> float:
    # obtain the total memory occupied by torch.Tensor in the garbage collector
    # (units in MiB)

    # obtaining all the tensor objects from the garbage collector
    tensor_objs = [obj for obj in gc.get_objects() if isinstance(obj, torch.Tensor)]

    # iterate each tensor objects uniquely and calculate the total storage
    visited_data = set([])
    total_mem = 0.0
    for tensor in tensor_objs:
        if tensor.is_sparse:
            continue

        # check if it has been visited
        storage = tensor.storage()
        data_ptr = storage.data_ptr()  # type: ignore
        if data_ptr in visited_data:
            continue
        visited_data.add(data_ptr)

        # calculate the storage occupied
        numel = storage.size()
        elmt_size = storage.element_size()
        mem = numel * elmt_size / (1024 * 1024)  # in MiB

        total_mem += mem

    return total_mem
