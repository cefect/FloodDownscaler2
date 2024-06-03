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
import parameters
 
 
from ...whiteboxtools.whitebox_tools import WhiteboxTools
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
 