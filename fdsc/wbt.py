'''
Created on Dec. 5, 2022

@author: cefect
'''

import os, logging, sys
from whitebox_tools import WhiteboxTools
from definitions import wbt_dir
from fdsc.hp.oop import Basic
import config
 

class WBT_worker(Basic, WhiteboxTools):
    """init the WhiteboxTools to this project"""
    def __init__(self, 
                 build_dir=None,
                 logger=None,
                 compress_rasters=False,
                 **kwargs):
        
        WhiteboxTools.__init__(self)
        
        #setup logger
        """requires  a logger for the callback method"""
        if logger is None:
            #basic standalone setup
            logging.basicConfig(force=True, #overwrite root handlers
                stream=sys.stdout, #send to stdout (supports colors)
                level=logging.INFO, #lowest level to display
                )
            logger = logging.getLogger()
 
        
        
        Basic.__init__(self, logger=logger, **kwargs)
        
        #=======================================================================
        # customizet wbt
        #=======================================================================
        #set the whitebox dir
        if build_dir is None:
            build_dir = wbt_dir
            
        assert os.path.exists(build_dir), 'bad directory for whitebox-tools... check definitions.py wbt_dir'
        self.set_whitebox_dir(build_dir)
        #print(f'set_whitebox_dir({build_dir})')
        
        #callback default
        self.set_default_callback(self.__callback__)
        
        #verbosity
        if not config.log_level==logging.INFO:
            self.set_verbose_mode(True)
        else:
            self.set_verbose_mode(False)
            
        if not self.set_compress_rasters(compress_rasters)==0:
            raise IOError('set_compress_rasters')
            

        
        #=======================================================================
        # wrap
        #=======================================================================
        self.logger.debug('setup WhiteBoxTools w/\n' +\
                 
                 f'    set_whitebox_dir({build_dir})\n'+\
                 f'    set_verbose_mode({__debug__})\n'+\
                 f'    set_compress_rasters({compress_rasters})\n'+\
                 f'    set_default_callback(self.__callback__)'
                 #"Version information: {}".format(self.version())                
                 )
                 
        
    def __callback__(self, value):
        """default callback methjod"""
        if not "%" in value:
            self.logger.debug(value)
        
            
        
        
        
        