'''
Created on Mar 5, 2019

@author: cef

 
'''



#===============================================================================
# # imports --------------------------------------------------------------------
#===============================================================================
import numpy as np
import warnings

np.set_printoptions(linewidth=200)
 
  
def apply_block_reduce2(ar1, aggscale=2, func=np.mean):
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
    
#===============================================================================
#     
# 
# def downsample(a, n=2):
#     """increase shape. scale up an array by replicating parent cells onto children with spatial awareness
#     
#     Parameters
#     ----------
#     n: int, default 2
#         amount to scale shape by
#     
#     very confusing.. surprised there is no builtin"""
#     
#     
#     
#     assert isinstance(n, int)
#     assert n>1
#     new_shape = (a.shape[0]*n, a.shape[1]*n)
#     
#     """runs out of memory
#     #===========================================================================
#     # np.kron
#     #===========================================================================
#     
#     np.kron(a, np.ones((n,n)))"""
#     
#     """interploates
#     #===========================================================================
#     # scipy.ndimage.zoom
#     #===========================================================================
#     zoom(a, n, """
#     
#     #===========================================================================
#     # scipy.ndimage.zoom
#     #===========================================================================
#     """preferred method"""
#     raise IOError('use ndimage.zoom in place')
#     scipy.ndimage.zoom(mar_raw, scale, order=0, mode='reflect',   grid_mode=True)
#     
#     #===========================================================================
#     # skimage.transform.resize
#     #===========================================================================
#  #==============================================================================
#  #    """seems to work.. should be a faster way though w/o polynomial"""
#  #    
#  #    res_ar2 = skimage.transform.resize(a, new_shape, order=0, mode='constant')
#  #    res_ar = res_ar2
#  #    
#  #    """tiles blocks... doesn't zoom
#  #    #===========================================================================
#  #    # np.tile
#  #    #===========================================================================
#  #    #np.tile(np.tile(a, n).T, a.shape[0]//downscale)"""
#  #    
#  #    """
#  #    #===========================================================================
#  #    # concat list
#  #    #===========================================================================
#  #    
#  #    l=list()
#  #    for i in range(a.shape[0]):
#  #        #=======================================================================
#  #        # l = list()
#  #        # for b in build_blocks(a[i, :].reshape(-1), n=n):
#  #        #     l.append(b)
#  #        #=======================================================================
#  # 
#  #        
#  #        new_ar = np.concatenate([b for b in build_blocks(a[i, :].reshape(-1), n=n)], axis=1)
#  #        #print('i=%i\n%s'%(i, new_ar))
#  #        l.append(new_ar)
#  #    
#  #    res_ar = np.concatenate(l, axis=0) 
#  #    assert np.array_equal(res_ar2, res_ar)
#  #    """
#  #    
#  #    assert res_ar.shape==new_shape
#  #    
#  #    return res_ar
#  #==============================================================================
# 
# 
# #===============================================================================
# # def xxxupsample2(a, n=2):
# #     """scale up an array by replicating parent cells onto children with spatial awareness
# #     
# #     using apply
# #     
# #     this is slower!"""
# #     
# #     
# #     
# #     def row_builder(a1, n=2):
# #         row_ar = np.concatenate([b for b in build_blocks(a1, n=n)], axis=1)
# #         return row_ar
# #  
# #     """only useful for 3d it seems
# #     np.apply_over_axes(row_builder, a, [0,1])"""
# #     
# #     #build blocks for each row (results are stacked as in 3D)
# #     res3d_ar = np.apply_along_axis(row_builder, 1, a, n=n)
# #     
# #     #split out each 3D and recombine horizontally
# #     res_ar = np.hstack(np.split(res3d_ar, res3d_ar.shape[0], axis=0))[0]
# #  
# #     #check
# #     new_shape = tuple([int(e) for e in np.fix(np.array(a.shape)*n).tolist()])
# #     assert res_ar.shape==new_shape
# #     return res_ar
# #         
# #      
# # 
# # def build_blocks(a, n=2):
# #     """generate 2D blocks"""
# #     for x in np.nditer(a):
# #         yield np.full((n,n), x)
# #===============================================================================
#  
#===============================================================================

def get_support_ratio(ar_top, ar_bot):
        """get scale difference"""
        shape1 = ar_top.shape
        shape2 = ar_bot.shape
        
        height_ratio = shape1[0]/shape2[0]
        width_ratio = shape1[1]/shape2[1]
        
        assert height_ratio==width_ratio, f'ratio mismatch. height={height_ratio}. width={width_ratio}'
        
        return width_ratio
