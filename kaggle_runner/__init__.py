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
    try:
        print("TRY pytest set_trace")
        pytest.set_trace()
    except Exception as e:
        print(e)
        print("TRY pytest set_trace failed")
        import pytest
        pytest.set_trace()
        pdb.set_trace()

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
