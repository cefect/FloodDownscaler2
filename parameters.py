"""WARNING: this should be superceeded if FlooDownscaler2 is added as a submodule"""
import os

import logging, datetime
log_level = logging.DEBUG

#===============================================================================
# directories
#===============================================================================
src_dir = os.path.dirname(os.path.abspath(__file__))
src_name = os.path.basename(src_dir)

#location of logging configuration file
logcfg_file=os.path.join(src_dir, r'fdsc\hp\logger.conf')


#===============================================================================
# autos
#===============================================================================
today_str = datetime.datetime.now().strftime("%Y%m%d")