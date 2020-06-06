import subprocess
import os
import sys
from importlib import reload
__all__ = ["may_debug", "logger", "runners.runner",
           "runners.coordinator", "utils"]

from .defaults import DEBUG, RIPDB
from .utils import logger, AMQPURL
from .runners import coordinator

logger = logger

def reload_me():
    current_module = sys.modules[__name__]
    reload(current_module)

def may_debug(force=False):
    # import rpdb
    subprocess.run('cd /kaggle/working; '
        'git stash; git pull; git submodule update --init --recursive', shell=True, check=True)
    reload_me()

    import pdb

    if force:
        pdb.set_trace()
    else:
        if DEBUG:
            pdb.set_trace()

if __name__ == "__main__":
    port = sys.argv[1]
    assert int(port) >= 0
    phase = sys.argv[2]
    logger.debug(f"Paramters for creating runner: {sys.argv}")
    tmp_path = '.r'

    subprocess.run(f"rm -rf {tmp_path}", shell=True, check=True)
    coord = coordinator.Coordinator(tmp_path, "Test Runner")
    config = {"phase": phase, "port":port, "size": 384, "network": "intercept",
              "AMQPURL": AMQPURL()}
    path = coord.create_runner(config, 19999)

    if os.getenv("CI") != "true":
        ret = coord.push(path)  # just push first
        assert ret.returncode == 0
