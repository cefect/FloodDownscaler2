'''
Created on May 25, 2024

@author: cef
'''

#===============================================================================
# imports----------
#===============================================================================
import os, argparse, logging, datetime
import rioxarray
from rasterio.enums import Resampling

from scipy.ndimage import distance_transform_edt
from skimage.feature import peak_local_max
from skimage.graph import MCP_Geometric
#from osgeo import gdal

from parameters import today_str
from fdsc.hp.logr import get_new_file_logger, get_log_stream
from fdsc.hp.xr import resample_match_xr, dataarray_from_masked
from fdsc.assertions import *

from .coms import shape_ratio



 
    
def _cost_distance_fill(mar, cost_ar):
    """
    Fills masked values in a 2D array using cost-distance weighted nearest neighbor interpolation.

     
    could use scipy.ndimage.distance_transform_edt to create a max_dist buffer
        reduce the number of sources/targets as we're only concerned with edges
     
    """
    assert isinstance(mar, ma.MaskedArray)
    assert isinstance(cost_ar, np.ndarray)
    
 
    # Create MCP object with cost surface
    mcp = MCP_Geometric(cost_ar,
                        fully_connected=False, #d4
                        sampling=None, #square grid
                        )
    
    end_indices = np.argwhere(mar.mask)  # Coordinates of masked cells
    
    # Find least-cost paths from all masked points to unmasked points
    cumulative_costs, traceback_ar = mcp.find_costs(
        starts=np.argwhere(~mar.mask),
        #ends=end_indices,
        #find_all_ends=True, #minimum-cost-path to every specified end-position will be found;
        )
    
    """just shows the directions available for the distance cals
        offsets = np.array([
            [-1, 0],   # Move left
            [1, 0],    # Move right
            [0, -1],   # Move up
            [0, 1]     # Move down
        ])"""
    # Prepare for Filling
    filled_ar = mar.data.copy() 
    
    
    # Fill Masked Values
 #==============================================================================
 #    for end_idx in end_indices:
 #        i, j = end_idx  # Unpack coordinates directly
 #        tb = mcp.traceback(end_idx)
 #        start, end = tb[0], tb[1]
 # 
 #        filled_ar[i, j] =  mar.data[start]
 #==============================================================================
    # Fill Masked Values (List Comprehension Version)
    filled_ar[tuple(zip(*end_indices))] = [mar.data[mcp.traceback(idx)[0]] for idx in end_indices]
    
    return filled_ar

 


def downscale_costGrow_xr(dem_fine_xr, wse_coarse_xr,
                          
                cost='neutral',
                          
                 logger=None,
                 write_meta=True,
                 debug=__debug__,
                 ):
    """
    downscale a coarse WSE grid using costGrow algos and xarray
    
    params
    --------
    cost: str
        type of cost surface to use
            neutral: neutral cost surface
    """
    
    #=======================================================================
    # defaults
    #=======================================================================
    log = logger.getChild('costGrow')
    
    meta_d=dict()
    nodata = dem_fine_xr.rio.nodata
    #===========================================================================
    # pre-chescks
    #===========================================================================
    if debug:
        assert_dem_xr(dem_fine_xr, msg='DEM')
        assert_wse_xr(wse_coarse_xr, msg='WSE')        
        assert_equal_extents_xr(dem_fine_xr, wse_coarse_xr, msg='\nraw inputs')
        
 
    #===========================================================================
    # get rescaling value
    #===========================================================================
    shape_rat_t = shape_ratio(dem_fine_xr, wse_coarse_xr)
 
    assert_integer_like_and_nearly_identical(np.array(shape_rat_t))
    
    downscale=int(shape_rat_t[0])
    
    #===========================================================================
    # finish init
    #===========================================================================
    if write_meta:
        wse_mask = wse_coarse_xr.to_masked_array().mask
        meta_d.update(
            {'downscale':downscale,
                'fine_shape':dem_fine_xr.shape,
             'coarse_shape':wse_coarse_xr.shape,
             'start':datetime.datetime.now(),
             'dem_mask_cnt':dem_fine_xr.to_masked_array().mask.sum(),
             'wse_wet_cnt':np.invert(wse_mask).sum(),
             'debug':debug,
             })
        
    log.info(f'passed all checks and downscale={downscale}\n    {meta_d}')
    
 
    
    #===========================================================================
    # 01 resample
    #===========================================================================
    wse_fine_xr1 = resample_match_xr(wse_coarse_xr, dem_fine_xr, resampling=Resampling.bilinear)
    
    if write_meta:
        meta_d['wse_fine_1resap_wet_cnt']=np.invert(wse_fine_xr1.to_masked_array().mask).sum()
        
    if debug:
        assert_wse_xr(wse_fine_xr1)
        assert_equal_raster_metadata(wse_fine_xr1, dem_fine_xr, msg='post-resampling')

    #===========================================================================
    # 02 wet partials
    #===========================================================================
    wse_mar = wse_fine_xr1.to_masked_array()
    dem_mar = dem_fine_xr.to_masked_array()
    
    wse_mar2 = ma.MaskedArray(np.nan_to_num(wse_mar.data, nodata),
              mask=np.logical_or(
                  np.logical_or(dem_mar.mask, wse_mar.mask), #union of masks
                  wse_mar<=dem_mar, #below ground water
                  ))
    
    wse_fine_xr2 = dataarray_from_masked(wse_mar2, wse_fine_xr1)
    
    if write_meta:
        meta_d['wse_fine_2wp_wet_cnt']=np.invert(wse_fine_xr2.to_masked_array().mask).sum()
        
        
    if debug:
        if not wse_mar.mask.sum()<=wse_mar2.mask.sum():
            raise AssertionError('expected wet-cell count to decrease during wet-partial treatment')
        
        
    #===========================================================================
    # 03 dry partials
    #===========================================================================
    wse_mar3 = wse_fine_xr2.to_masked_array()
    if cost=='neutral':
        """probably something faster if we're using a neutral surface"""
        cost_ar = np.ones(wse_mar3.shape)        
        wse_filled_ar = _cost_distance_fill(wse_mar3, cost_ar)
        
    elif cost=='terrain':
        #compute neutral wse impute
        
        #get delta
        
        #normalize cost surface
        """
        from zero to one?
            no... want more separation between positive/negative
        negative:free; positives:some water surface slope tuning?
            should read more about what the cost surface is
        """
                
 
    else:
        raise KeyError(cost)
    
    wse_fine_xr3 = dataarray_from_masked(ma.MaskedArray(wse_filled_ar, dem_mar.mask), wse_fine_xr2)
 
 
    
    
        

    
    
 