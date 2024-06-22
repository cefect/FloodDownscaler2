'''
Created on May 26, 2024

@author: cef
'''

#===============================================================================
# import sys, os
# print(sys.executable)
# print(f'sys.version: {sys.version}')
# print(f'os.getcwd(): {os.getcwd()}')
# print('sys.path\n' + '\n'.join(sys.path))
# print('PYTHONPATH')
# if 'PYTHONPATH' in os.environ: print('\n'.join(os.environ['PYTHONPATH'].split(';')))
#===============================================================================


import logging, os
from ..parameters import log_level
 
from ...whiteboxtools.whitebox_tools import WhiteboxTools
wbt = WhiteboxTools()

from definitions import wbt_dir, wrk_dir
wbt.set_whitebox_dir(wbt_dir)
wbt.set_working_dir(wrk_dir) 
wbt.set_compress_rasters(True)
wbt.set_max_procs(1) #needed by HPC?
#===============================================================================
# configure logging
#===============================================================================
if log_level>=logging.INFO:
    wbt.set_verbose_mode(False)
else:
    wbt.set_verbose_mode(True)
    #print(f'WhiteBoxTools initated w/ \n{wbt.version()}')

"""not working for some reason... set excplicitly in the function call
logger = logging.getLogger('wbt')


wbt.set_default_callback(lambda value: logger.debug(value) if not "%" in value else None)

"""
 
 
class WhiteBoxToolsCallFail(Exception):
    """Exception raised for errors in the execution of WhiteBoxTools."""
    pass
