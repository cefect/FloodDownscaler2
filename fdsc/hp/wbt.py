'''
Created on May 26, 2024

@author: cef
'''
import logging
import parameters
from ...whitebox_tools.whitebox_tools import WhiteboxTools
wbt = WhiteboxTools()

from definitions import wbt_dir, wrk_dir
wbt.set_whitebox_dir(wbt_dir)
wbt.set_working_dir(wrk_dir)
 
wbt.set_compress_rasters(True)
#===============================================================================
# configure logging
#===============================================================================
if parameters.log_level>=logging.INFO:
    wbt.set_verbose_mode(False)
else:
    wbt.set_verbose_mode(True)

"""not working for some reason... set excplicitly in the function call
logger = logging.getLogger('wbt')


wbt.set_default_callback(lambda value: logger.debug(value) if not "%" in value else None)

"""
 