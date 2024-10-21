'''
Created on Jun. 3, 2024

@author: cef
'''

import numpy as np
 
def apply_block_reduce(ar1, aggscale=2, func=np.mean):
    """apply a reducing function to a numpy array in 2d blocks
    
    see apply_block_reduce() for notes"""
    
 
    # get teh new shape
 
    assert isinstance(aggscale, int)
    assert aggscale>1    
    new_shape = (ar1.shape[0]//aggscale, ar1.shape[1]//aggscale)
    
    
    #stack windows into axis 1 and 3
    ar1_stacked = ar1.reshape(ar1.shape[0]//aggscale, aggscale, ar1.shape[1]//aggscale, aggscale)
    
    ##apply function
    res_ar2 = func(ar1_stacked, axis=(1,3))
    
    #check
    assert res_ar2.shape==new_shape
    
    return res_ar2