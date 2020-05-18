__all__ = ["runners.runner", "runners.coordinator", "utils"]

import ripdb

from .defaults import RIPDB
from .utils import logger

__port = 4444

def may_debug():
    global __port

    if RIPDB:
        try:
            logger.debug(f"try port %s for ripdb", __port)
            ripdb.set_trace(port=__port)
        except OSError:
            __port += 1
            logger.debug(f"try port %s for ripdb", __port)
            ripdb.set_trace(port=__port)
        else:
            __port += 1
        finally:
            logger.debug(f"port %s for ripdb done", __port-1)
