import subprocess
import os
import sys
from .utils import logger, AMQPURL
from .runners import coordinator

if __name__ == "__main__":
    port = sys.argv[1]
    assert int(port) >= 0
    phase = sys.argv[2]
    logger.debug(f"Paramters for creating runner: {sys.argv}")
    tmp_path = '.r'

    subprocess.run(f"rm -rf {tmp_path}", shell=True, check=True)
    coord = coordinator.Coordinator("Test Runner", tmp_path)
    config = {"phase": phase, "port":port, "size": 384, "network": "intercept",
              "AMQPURL": AMQPURL()}
    path = coord.create_runner(config, 19999, script=False)

    if os.getenv("CI") != "true":
        ret = coord.push(path)  # just push first
        assert ret.returncode == 0
