import subprocess
import os
import sys
from importlib import reload
__all__ = ["may_debug", "logger", "runners.runner",
           "runners.coordinator", "utils"]

from .defaults import DEBUG, RIPDB, INTERACTIVE

from .utils import logger

logger = logger

def reload_me():
    current_module = sys.modules[__name__]
    reload(current_module)

def may_debug(force=False):
    subprocess.run('cd /kaggle/working; '
        'git stash; git pull; git submodule update --init --recursive', shell=True, check=True)
    reload_me()

    import pdb

    if force:
        if "pytest" in sys.modules:
            pytest.set_trace()
        else:
            if INTERACTIVE:
                import ipdb
                ipdb.set_trace()
            else:
                pdb.set_trace()
    elif DEBUG:
        if "pytest" in sys.modules:
            pytest.set_trace()
        else:
            if INTERACTIVE:
                import ipdb
                ipdb.set_trace()
            else:
                pdb.set_trace()
