import subprocess
import os
import sys
__all__ = ["may_debug", "logger", "runners.runner",
           "runners.coordinator", "utils"]

from .defaults import DEBUG, RIPDB, INTERACTIVE

from .utils import logger

logger = logger

def reload_me():
    from importlib import reload
    current_module = sys.modules[__name__]
    reload(current_module)

def may_debug(force=False):
    subprocess.run('cd /kaggle/working || cd kaggle_runner; '
        'git commit -asm "Good game"; git pull; git submodule update --init --recursive', shell=True)
    reload_me()

    import pdb

    if force:
        if "pytest" in sys.modules:
            import pytest
            pytest.set_trace()
        else:
            if INTERACTIVE:
                import ipdb
                ipdb.set_trace()
            else:
                pdb.set_trace()
    elif DEBUG:
        if "pytest" in sys.modules:
            import pytest
            pytest.set_trace()
        else:
            if INTERACTIVE:
                import ipdb
                ipdb.set_trace()
            else:
                pdb.set_trace()
