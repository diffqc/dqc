from __future__ import annotations
import contextlib
from typing import Optional, List, Callable
import torch
import numpy as np
import h5py

class Cache(object):
    def __init__(self):
        self._cacheable_pnames: List[str] = []
        self._fname: Optional[str] = None
        self._pnames_to_cache: Optional[List[str]] = None
        self._fhandler: Optional[h5py.File] = None

    def set(self, fname: str, pnames: Optional[List[str]] = None):
        # set up the cache
        self._fname = fname
        self._pnames_to_cache = pnames

    def cache(self, pname: str, fcn: Callable[[], torch.Tensor]) -> torch.Tensor:
        # if not has been set, then just calculate and return
        if not self._isset():
            return fcn()

        # if the param is not to be cached, then just calculate and return
        if not self._pname_to_cache(pname):
            return fcn()

        # get the dataset name
        dset_name = self._pname2dsetname(pname)

        # check if dataset name is in the file (cache exist)
        file = self._get_file_handler()
        if dset_name in file:
            # retrieve the dataset if it has been computed before
            res = self._load_dset(dset_name, file)
        else:
            # if not in the file, then compute the tensor and save the cache
            res = fcn()
            self._save_dset(dset_name, file, res)
        return res

    @contextlib.contextmanager
    def open(self):
        # open the cache file
        try:
            if self._fname is not None:
                self._fhandler = h5py.File(self._fname, "a")
            yield self._fhandler
        finally:
            if self._fname is not None:
                self._fhandler.close()
                self._fhandler = None

    def add_prefix(self, prefix: str) -> Cache:
        # return a Cache object that will add the prefix for every input of
        # parameter names
        prefix = _normalize_prefix(prefix)
        return _PrefixedCache(self, prefix)

    def add_cacheable_params(self, pnames: List[str]):
        # add the cacheable parameter names
        self._cacheable_pnames.extend(pnames)

    def get_cacheable_params(self) -> List[str]:
        # return the cacheable parameter names
        return self._cacheable_pnames

    @staticmethod
    def get_dummy() -> Cache:
        # returns a dummy cache that does not do anything
        return _DummyCache()

    def _pname_to_cache(self, pname: str) -> bool:
        # check if the input parameter name is to be cached
        return (self._pnames_to_cache is None) or (pname in self._pnames_to_cache)

    def _pname2dsetname(self, pname: str) -> str:
        # convert the parameter name to dataset name
        return pname.replace(".", "/")

    def _get_file_handler(self) -> h5py.File:
        # return the file handler, if the file is not opened yet, then raise an error
        if self._fhandler is None:
            msg = "The cache file has not been opened yet, please use .open() before reading/writing to the cache"
            raise RuntimeError(msg)
        else:
            return self._fhandler

    def _isset(self) -> bool:
        # returns the indicator whether the cache object has been set
        return self._fname is not None

    def _load_dset(self, dset_name: str, fhandler: h5py.File) -> torch.Tensor:
        # load the dataset from the file handler (check is performed outside)
        dset_np = np.asarray(fhandler[dset_name])
        dset = torch.as_tensor(dset_np)
        return dset

    def _save_dset(self, dset_name: str, fhandler: h5py.File, dset: torch.Tensor):
        # save res to the h5py in the dataset name
        fhandler[dset_name] = dset.detach()

class _PrefixedCache(Cache):
    # this class adds a prefix to every parameter names input
    def __init__(self, obj: Cache, prefix: str):
        self._obj = obj
        self._prefix = prefix

    def set(self, fname: str, pnames: Optional[List[str]] = None):
        # set must only be done in the parent object, not in the children objects
        raise RuntimeError("Cache.set() must be done on non-prefixed cache")

    def cache(self, pname: str, fcn: Callable[[], torch.Tensor]) -> torch.Tensor:
        return self._obj.cache(self._prefixed(pname), fcn)

    @contextlib.contextmanager
    def open(self):
        try:
            with self._obj.open() as f:
                yield f
        finally:
            pass

    def add_prefix(self, prefix: str) -> Cache:
        # return a deeper prefixed object
        prefix = self._prefixed(_normalize_prefix(prefix))
        return self._obj.add_prefix(prefix)

    def add_cacheable_params(self, pnames: List[str]):
        # add the cacheable parameter names
        pnames = [self._prefixed(pname) for pname in pnames]
        self._obj.add_cacheable_params(pnames)

    def get_cacheable_params(self) -> List[str]:
        # this can only be done on the root cache (non-prefixed) to avoid
        # confusion about which name should be provided (absolute or relative)
        raise RuntimeError("Cache.get_cacheable_params() must be done on non-prefixed cache")

    def _prefixed(self, pname: str) -> str:
        # returns the prefixed name
        return self._prefix + pname

class _DummyCache(Cache):
    # this class just an interface of cache without doing anything
    def __init__(self):
        pass

    def set(self, fname: str, pnames: Optional[List[str]] = None):
        pass

    def cache(self, pname: str, fcn: Callable[[], torch.Tensor]) -> torch.Tensor:
        return fcn()

    @contextlib.contextmanager
    def open(self):
        try:
            yield None
        finally:
            pass

    def add_prefix(self, prefix: str) -> Cache:
        # return a deeper prefixed object
        return self

    def add_cacheable_params(self, pnames: List[str]):
        pass

    def get_cacheable_params(self) -> List[str]:
        return []

def _normalize_prefix(prefix: str) -> str:
    # added a dot at the end of prefix if it is not so
    if not prefix.endswith("."):
        prefix = prefix + "."
    return prefix
