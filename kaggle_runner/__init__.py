import subprocess
import os
import sys
from importlib import reload
__all__ = ["may_debug", "logger", "runners.runner",
           "runners.coordinator", "utils"]

from .defaults import DEBUG, RIPDB

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
        pdb.set_trace()
    else:
        if DEBUG:
            pdb.set_trace()
