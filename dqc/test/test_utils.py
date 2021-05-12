from dqc.utils.config import config
from dqc.utils.misc import logger

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
