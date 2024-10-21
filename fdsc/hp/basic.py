'''
Created on Jun. 28, 2024

@author: cef
'''

import os, logging, pprint, webbrowser, sys
import logging.config
from datetime import datetime

today_str = datetime.now().strftime("%Y%m%d")


class LoggedException(Exception):
    """Custom exception that logs its message using the provided logger."""
    
    def __init__(self, message, log=None):
        super().__init__(message)
        if not log is None:
            log.error(message)