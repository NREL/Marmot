

import logging
from marmot.formatters.formatbase import Process

logger = logging.getLogger('formatter.'+__name__)


class ProcessEGRET(Process):
    """Process EGRET class specific data from a json database.
    """
    # Maps EGRET property names to Marmot names, 
    # unchanged names not included  
    PROPERTY_MAPPING: dict = {}

    pass