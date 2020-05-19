__all__ = ["may_debug","logger", "runners.runner", "runners.coordinator", "utils"]

import ipdb

from .defaults import DEBUG, RIPDB
from .utils import logger

logger = logger

def may_debug():
    if DEBUG:
        ipdb.set_trace()
