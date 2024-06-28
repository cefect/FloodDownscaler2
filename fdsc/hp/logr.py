'''
Created on Mar. 26, 2020

@author: cefect

usually best to call this before any standard imports
    some modules have auto loggers to the root loger
    calling 'logging.getLogger()' after these configure will erase these
'''
import os, logging, logging.config, pprint, sys




plog = lambda msg, log: print(msg) if log is None else log.debug(msg)


        
def get_new_file_logger(
        logger_name='log',
        level=logging.DEBUG,
        fp=None, #file location to log to
        logger=None,
        mode='w',
        ):
    
    #===========================================================================
    # configure the logger
    #===========================================================================
    if logger is None:
        logger = logging.getLogger(logger_name)
        
    logger.setLevel(level)
    
    #===========================================================================
    # configure the handler
    #===========================================================================
    assert fp.endswith('.log')
    
    formatter = logging.Formatter('%(levelname)s.%(asctime)s.%(name)s:  %(message)s',
                                  datefmt='%H:%M:%S')        
    handler = logging.FileHandler(fp, mode=mode) #Create a file handler at the passed filename 
    handler.setFormatter(formatter) #attach teh formater object
    handler.setLevel(level) #set the level of the handler
    
    logger.addHandler(handler) #attach teh handler to the logger
    
    logger.debug('built new file logger  here \n    %s'%(fp))
    
    return logger
    
    
def get_log_stream(
        logger_name='log',
        level=logging.DEBUG,
        logger=None):
    
    #===========================================================================
    # configure the logger
    #===========================================================================
    if logger is None:
        logger = logging.getLogger(logger_name)
        
    logger.setLevel(level)
    
    #===========================================================================
    # check if the logger already has a StreamHandler
    #===========================================================================
    has_stream_handler = any(isinstance(handler, logging.StreamHandler) for handler in logger.handlers)
    
    if not has_stream_handler:
        #===========================================================================
        # configure the handler
        #===========================================================================
        formatter = logging.Formatter('%(levelname)s.%(name)s:  %(message)s')        
        handler = logging.StreamHandler(
            stream=sys.stdout,  # send to stdout (supports colors)
        )  
        handler.setFormatter(formatter)  # attach the formatter object
        handler.setLevel(level)  # set the level of the handler
        
        logger.addHandler(handler)  # attach the handler to the logger
        logger.debug('Built new console logger')
    else:
        pass
        #logger.debug('Console logger already exists')

    return logger
    
    
    
    
    
    
    
    