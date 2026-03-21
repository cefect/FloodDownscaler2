'''
Created on May 26, 2024

@author: cef
'''
import logging, os, multiprocessing, subprocess
import warnings

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
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
current_wdir = os.getcwd() #WBT moves htis

from ..parameters import log_level

"""WARNING... something here changes the directory?"""
from whitebox import WhiteboxTools
wbt = WhiteboxTools()

# resolve WhiteboxTools executable directory from environment with package fallback
default_wbt_dir = wbt.exe_path
resolved_wbt_dir = default_wbt_dir
configured_wbt_dir = None
configured_wbt_key = None
for env_key in ("FDSC_WBT_DIR", "WHITEBOXTOOLS_DIR", "WBT_DIR"):
    env_value = os.environ.get(env_key)
    if env_value:
        configured_wbt_dir = os.path.abspath(os.path.expanduser(env_value))
        configured_wbt_key = env_key
        break

if configured_wbt_dir:
    configured_wbt_fp = configured_wbt_dir
    if os.path.isdir(configured_wbt_dir):
        configured_wbt_fp = os.path.join(configured_wbt_dir, wbt.exe_name)

    if os.path.isfile(configured_wbt_fp):
        resolved_wbt_dir = os.path.dirname(configured_wbt_fp)
    else:
        warnings.warn(
            f"invalid ${configured_wbt_key} for {wbt.exe_name}. expected executable at \n    {configured_wbt_fp}\n"
            f"using package default at \n    {default_wbt_dir}"
        )

resolved_wbt_fp = os.path.join(resolved_wbt_dir, wbt.exe_name)
if not os.path.isfile(resolved_wbt_fp):
    raise FileNotFoundError(
        f"WhiteboxTools executable not found: \n    {resolved_wbt_fp}\n"
        f"set FDSC_WBT_DIR to a directory or full path containing {wbt.exe_name}"
    )
# if not os.path.exists(wrk_dir):
#     warnings.warn(f'Working directory not found: {wrk_dir}')
wbt.set_whitebox_dir(resolved_wbt_dir)
wbt.set_working_dir(project_root) 
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
