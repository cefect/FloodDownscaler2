'''
Created on Mar 11, 2019

@author: cef

pandas functions 

py3.7

pd.__version__

'''
import logging
import numpy as np
import pandas as pd
import pandas.testing as pdt
import warnings

#===============================================================================
# pandas global options
#===============================================================================
pd.options.mode.chained_assignment = None   #setting with copy warning handling

#truncate thresholds
#===============================================================================
# pd.set_option("display.max_rows", 20)
# 
# pd.set_option("display.max_columns", 10)
# pd.set_option("display.max_colwidth", 12)
# 
# #truncated views
# pd.set_option("display.min_rows", 15)
# pd.set_option("display.min_rows", 15)
# pd.set_option('display.width', 150)
#===============================================================================

 
#===============================================================================
# custom imports
#===============================================================================

#from hp.exceptions import Error
#import hp.np
#from hp.np import left_in_right as linr



#mod_logger = logging.getLogger(__name__) #creates a child logger of the root

bool_strs = {'False':False,
             'false':False,
             'FALSE':False,
             0:False,
             'True':True,
             'TRUE':True,
             'true':True,
             1:True,
             False:False,
             True:True}


#===============================================================================
#VIEWS ---------------------------------------------------------
#===============================================================================



def view_web_df(df):
    if isinstance(df, pd.Series):
        df = pd.DataFrame(df)
    import webbrowser
    #import pandas as pd
    from tempfile import NamedTemporaryFile

    with NamedTemporaryFile(delete=False, suffix='.html', mode='w') as f:
        #type(f)
        df.to_html(buf=f)
        
    webbrowser.open(f.name)
    
def view(df):
    view_web_df(df)
    
  
 
