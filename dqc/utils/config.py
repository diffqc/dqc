from dataclasses import dataclass

__all__ = ["config"]

@dataclass
class _Config(object):
    # Threshold memory (matrix above this size should not be constructed)
    THRESHOLD_MEMORY: int = 10 * 1024 ** 3  # in B

config = _Config()
