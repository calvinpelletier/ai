import os
import pytest

from ai.util.testing import *
from ai.examples.imsim.main import CLI, LOSSES


@pytest.mark.filterwarnings('ignore:.*deprecated')
def test_imsim():
    os.environ['AI_LAB_PATH'] = '/tmp/testing/lab'
    cli = CLI()
    cli.clean()
    cli.hps(n=1, steplimit=2, prune=False)
    for loss in LOSSES:
        cli.train(loss, steplimit=4)
    cli.compare()
