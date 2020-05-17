__all__ = ["runners.runner", "runners.coordinator", "utils"]

import ripdb
from utils import logger

from .defaults import RIPDB

__port = 4444

def may_debug():
    global __port

    if RIPDB:
        try:
            ripdb.set_trace(port=__port)
        except OSError:
            __port += 1
            ripdb.set_trace(port=__port)
        else:
            __port += 1
        finally:
            logger.debug(f"{__port} for ripdb")
