__all__ = ["runners.runner", "runners.coordinator", "utils"]

import ipdb

from .defaults import DEBUG, RIPDB
from .utils import logger

__port = 4444
__rip = False

def may_debug():
    global __port
    global __rip

    # if RIPDB and not __rip:
    #     try:
    #         logger.debug(f"try port %s for ripdb", __port)
    #         ripdb.set_trace(port=__port)
    #     except OSError:
    #         __port += 1
    #         logger.debug(f"try port %s for ripdb", __port)
    #         ripdb.set_trace(port=__port)
    #     finally:
    #         __port += 1
    #         __rip = True
    #         logger.debug(f"port %s for ripdb finally", __port-1)
    # else:

    if DEBUG:
        ipdb.set_trace()
