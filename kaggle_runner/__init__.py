__all__ = ["may_debug", "logger", "runners.runner",
    "runners.coordinator", "utils"]

from .defaults import DEBUG, RIPDB
from .utils import logger

logger = logger


def may_debug(force=False):
    import rpdb
    import pdb

    if force:
        try:
            rpdb.set_trace()
        except:
            ...
            pdb.set_trace()
    else:
        if DEBUG:
            try:
                rpdb.set_trace()
            except:
                ...
                pdb.set_trace()
