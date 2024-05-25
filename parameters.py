"""WARNING: this should be superceeded if FlooDownscaler2 is added as a submodule"""
import os

import logging
log_level = logging.DEBUG

src_dir = os.path.dirname(os.path.abspath(__file__))
src_name = os.path.basename(src_dir)

#location of logging configuration file
logcfg_file=os.path.join(src_dir, r'fdsc\hp\logger.conf')