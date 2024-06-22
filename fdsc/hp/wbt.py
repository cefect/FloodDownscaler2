'''
Created on May 26, 2024

@author: cef
'''
import logging, os, multiprocessing, subprocess

#===============================================================================
# build info
#===============================================================================
#===============================================================================
# import sys, os
# print(sys.executable)
# print(f'sys.version: {sys.version}')
# print(f'os.getcwd(): {os.getcwd()}')
# print('sys.path\n' + '\n'.join(sys.path))
# print('PYTHONPATH')
# if 'PYTHONPATH' in os.environ: print('\n'.join(os.environ['PYTHONPATH'].split(';')))
#=============================================================================== 
#===============================================================================
# 
# # Method 1: Using the os module
# num_processors_os = os.cpu_count()
# print(f"Number of available processors (os module): {num_processors_os}")
# 
# # Method 2: Using the multiprocessing module
# num_processors_mp = multiprocessing.cpu_count()
# print(f"Number of available processors (multiprocessing module): {num_processors_mp}")
#===============================================================================

current_wdir = os.getcwd() #WBT moves htis

from ..parameters import log_level

"""WARNING... something here changes the directory?"""
 
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
    
os.chdir(current_wdir)

"""not working for some reason... set excplicitly in the function call
logger = logging.getLogger('wbt')


wbt.set_default_callback(lambda value: logger.debug(value) if not "%" in value else None)

"""
def wbt_subprocess(command, log=None, debug=False):
    """subprocess helper
    some systems dont play well with python implementation
    """
    
    if debug:
        command.append('-v')
    # Run the command using subprocess.Popen
    log.debug(command)
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    #log results
    stdout, stderr = process.communicate()
    log.debug(stdout.decode())
    if stderr:
        log.error(stderr.decode())
    # Check for errors
    if process.returncode != 0:
        raise Exception(f'fill_depressions failed: {stderr.decode()}')
    
 
class WhiteBoxToolsCallFail(Exception):
    """Exception raised for errors in the execution of WhiteBoxTools."""
    pass


