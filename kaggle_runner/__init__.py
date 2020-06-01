__all__ = ["may_debug","logger", "runners.runner", "runners.coordinator", "utils"]

from .defaults import DEBUG, RIPDB
from .utils import logger
import pdb

logger = logger

def may_debug(force=False):
    if force:
        pdb.set_trace()
    else:
        if DEBUG:
            pdb.set_trace()
