'''
Created on Oct. 23, 2023

@author: cefect
'''
import  os, tempfile, datetime

temp_dir = os.path.join(tempfile.gettempdir(), __name__, datetime.datetime.now().strftime('%Y%m%d'))
if not os.path.exists(temp_dir):
    os.makedirs(temp_dir)
    
    
    