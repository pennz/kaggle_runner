__all__ = ["runners.runner", "runners.coordinator", "utils"]

import ripdb

from .defaults import RIPDB
from .utils import logger

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
            logger.debug(f"port %s for ripdb", __port)
