import torch
import dqc.utils
from dqc.utils.config import config
from dqc.utils.misc import logger
from dqc.test.utils import assert_fail

def test_converter_length():
    a = torch.tensor([1.0])

    # convert to itself
    assert torch.allclose(dqc.utils.convert_length(a), a)
    # convert from atomic unit to angstrom
    assert torch.allclose(dqc.utils.convert_length(a, to_unit="angst"), a * 5.29177210903e-1)
    # convert from angstrom to atomic unit
    assert torch.allclose(dqc.utils.convert_length(a, from_unit="angst"), a / 5.29177210903e-1)
    # convert from angstrom to angstrom
    assert torch.allclose(dqc.utils.convert_length(a, from_unit="angst", to_unit="angst"), a)
    assert torch.allclose(dqc.utils.convert_length(a, from_unit="angst", to_unit="angstrom"), a)

def test_converter_wrong_unit():
    a = torch.tensor([1.0])
    assert_fail(lambda: dqc.utils.convert_length(a, from_unit="adsfa"), ValueError, ["'angst'"])

def test_logger(capsys):
    # test if logger behaves correctly
    s = "Hello world"
    logger.log(s)
    captured = capsys.readouterr()
    assert captured.out == ""

    config.VERBOSE = 1
    logger.log(s)
    captured = capsys.readouterr()
    assert captured.out == s + "\n"

    logger.log(s, vlevel=1)
    captured = capsys.readouterr()
    assert captured.out == ""

    # restore the verbosity level to 0
    config.VERBOSE = 0
