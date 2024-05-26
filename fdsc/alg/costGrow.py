'''
Created on May 25, 2024

@author: cef
'''

#===============================================================================
# imports----------
#===============================================================================
import os, argparse, logging, datetime, tempfile, pickle
import rioxarray
from rasterio.enums import Resampling
from tqdm.auto import tqdm

import scipy.ndimage 
import skimage.graph
#from osgeo import gdal

from parameters import today_str
from fdsc.hp.logr import get_new_file_logger, get_log_stream
from fdsc.hp.xr import resample_match_xr, dataarray_from_masked, xr_to_GeoTiff
from fdsc.assertions import *

from .coms import shape_ratio

#===============================================================================
# HELPERS------
#===============================================================================


def _get_od(out_dir):
    if out_dir is None:
        out_dir = tempfile.mkdtemp()
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        
    return out_dir
        
        
def _write_to_pick(data, fp, log=None):
    """writing pickels for tests"""
    
    plog = lambda msg: print(msg) if log is None else log.debug(msg)
    
    if not os.path.exists(os.path.dirname(fp)):
        os.makedirs(os.path.dirname(fp))
        
    with open(fp, "wb") as f:
        pickle.dump(data, f)
 
    plog(f'wrote to pickle\n    {fp}')
        
    
def _get_zero_padded_shape(shape):
    """
    Takes a shape-like tuple (e.g., (10, 10)) and returns a zero-padded string like '0010x0010'.

    Args:
        shape: A tuple representing the shape of a 2D array (e.g., (y_size, x_size)).

    Returns:
        str: A zero-padded string representing the 2D shape (e.g., '0010x0010').
    """

    try:
        # Ensure the input is a tuple with length 2
        if not isinstance(shape, tuple) or len(shape) != 2:
            raise ValueError("Input shape must be a tuple of length 2.")
        
        # Get the dimensions
        y_size, x_size = shape
        
        # Create a formatted string, zero-padding up to 4 digits
        shape_str = f"{y_size:04d}x{x_size:04d}" 
        
        return shape_str
    except ValueError as e:
        print(f"Error: {e}")
        return None  # Or raise the exception again, depending on your needs


def _distance_fill_cost_terrain(wse_fine_xr, dem_fine_xr, wse_coarse_xr, log=None,
                                cd_backend='wbt',out_dir=None, **kwargs
                                ):
    
 
    #===========================================================================
    # build cost surface from terrain
    #===========================================================================
    # #compute neutral wse impute
 
    #quick data fill on coarse, then resample
        
    wse_coarse_filled_ar = _distance_fill(wse_coarse_xr.to_masked_array(), log=log)
    wse_coarse_filled_xr = dataarray_from_masked(ma.MaskedArray(wse_coarse_filled_ar), wse_coarse_xr)
    
    log.debug(f'reseampling \'wse_coarse_filled_xr\' to fine')
    wse_filled_xr1 = resample_match_xr(wse_coarse_filled_xr, dem_fine_xr, resampling=Resampling.bilinear)
    
    
    #get delta
    log.debug(f'computing fine delta')
    delta_xr = dem_fine_xr - wse_filled_xr1
    
    #normalize cost surface
    """
    
    from zero to one?
    
        no... want more separation between positive/negative
    
    negative:free; positives:some water surface slope tuning?
    
        xr.where(delta_xr>0, 0.0, delta_xr)
    
        should read more about what the cost surface is
    
    """
    log.debug(f'computing cost surface')
    cost_xr = xr.where(delta_xr < 0, 0.0, delta_xr).fillna(999).round(1)
    cost_xr.rio.write_crs(dem_fine_xr.rio.crs, inplace=True) #not sure where this got lost
    
    #===========================================================================
    # #impute w/ cost
    #===========================================================================
    log.debug(f'imputing w/ costsurface and {cd_backend}')
    if cd_backend=='skimage':
        f = _distance_fill_cost_skimage
        
        f(wse_fine_xr.to_masked_array(), cost_xr.data, log=log, **kwargs)
        
        raise NotImplementedError('too slow')
        
    elif cd_backend=='wbt':
        #needs geospatial data
        
        wse_filled_xr = _distance_fill_cost_wbt(wse_fine_xr, cost_xr, log=log.getChild(cd_backend), 
                        out_dir = os.path.join(_get_od(out_dir), 'wbt'), **kwargs)
    else:
        raise KeyError(cd_backend)
 
    
    return wse_filled_xr

def _distance_fill_cost_skimage(mar, cost_ar, log=None):
    """
    Fills masked values in a 2D array using cost-distance weighted nearest neighbor interpolation.

    this is WAY too slow
    
    could use scipy.ndimage.distance_transform_edt to create a max_dist buffer
        reduce the number of sources/targets as we're only concerned with edges
     
    """
    assert isinstance(mar, ma.MaskedArray)
    assert mar.mask.any()
    assert not mar.mask.all()
    
    plog = lambda msg: print(msg) if log is None else log.debug(msg)
 
    # Create MCP object with cost surface
    plog(f'init on skimage.graph.MCP w/ {np.invert(mar.mask).sum()} source cells')
    mcp = skimage.graph.MCP(cost_ar,
                        fully_connected=False, #d4
                        sampling=None, #square grid
                        )
    
    end_indices = np.argwhere(mar.mask)  # Coordinates of masked cells
    
    # Find least-cost paths from all masked points to unmasked points
    plog(f'mcp.find_costs')
    cumulative_costs, traceback_ar = mcp.find_costs(
        starts=np.argwhere(~mar.mask),
        #ends=end_indices, #specifying these doesn't seem to improve performance
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
    plog(f'filling destination cells w/ least-cost sources')
    filled_ar[tuple(zip(*end_indices))] = [
        mar.data[mcp.traceback(idx)[0]] for idx in tqdm(end_indices, desc='distance_fill_cost')
        ]
    
    return filled_ar

def _distance_fill_cost_wbt(wse_xr, cost_xr, log=None, out_dir=None):
    """
    Fills masked values in a 2D array using cost-distance weighted nearest neighbor interpolation.
    """

    
    out_dir = _get_od(out_dir)
    
    to_gtif = lambda da, fn: xr_to_GeoTiff(da, os.path.join(out_dir, fn), log)
    
    
    #===========================================================================
    # init wbt
    #===========================================================================
    from fdsc.hp.wbt import wbt
    wbt.set_default_callback(lambda value: log.debug(value) if not "%" in value else None)
            
    #===========================================================================
    # prep
    #===========================================================================
    
    #add arbitrary increment to handle negative WSE
    add_increment=0
 
    if wse_xr.min()<0:
        add_increment=9999
        log.warning(f'adding increment {add_increment} to handle negative WSE')    
        wse_xr = wse_xr + add_increment
 
    
    #dump to GeoTiff
    log.debug(f'dumping Xarrays to GeoTiffs')
    wse_fp = to_gtif(wse_xr, '00_wse.tif')
    cost_fp = to_gtif(cost_xr, '00_cost.tif')
    
    
    #===========================================================================
    # #compute backlink raster
    #===========================================================================
    log.debug("wbt.cost_distance")
    backlink_fp = os.path.join(out_dir, f'01_backlink.tif')
    if not wbt.cost_distance(wse_fp,cost_fp, 
                             os.path.join(out_dir, f'01_outAccum.tif'),
                             backlink_fp) == 0:
        raise IOError('cost_distance')
    
    assert os.path.exists(backlink_fp)
    #=======================================================================
    # costAllocation
    #=======================================================================
    log.debug("wbt.cost_allocation")
    costAlloc_fp = os.path.join(out_dir, '02_costAllocation.tif')
    if not wbt.cost_allocation(wse_fp, backlink_fp, costAlloc_fp) == 0:
        raise IOError('wbt.cost_allocation')
    
    log.debug(f'wrote to \n    {costAlloc_fp}')
    #===========================================================================
    # back to Xarray
    #===========================================================================
    load_xr = lambda x: rioxarray.open_rasterio(x,masked=False).squeeze().rio.write_nodata(-9999)
    
    wse_filled_xr = load_xr(costAlloc_fp)
    wse_filled_xr.attrs = wse_xr.attrs.copy()
    
    log.debug('finished')
    
    return wse_filled_xr
 
 



def _distance_fill(mar,
                   method='distance_transform_cdt',
                   log=None, 
                   **kwargs):
    """fill masked values with their nearest unmasked value
    
    Params
    ------------
    method: str
        scipy.ndimage method with which to apply the distance calc
            distance_transform_cdt: chamfer transform
                fastest in 'case_f3n2e100', #EPSG:4326. 9000x9000, 3:1
                but not as nice looking
                
                takes an additional 'metric' kwarg
                    metric='chessboard', default
                        d8 (10% slower), a bit nicer looking
                    metric='taxicab'
                        d4
                
            distance_transform_edt: true-euclidian
                2x slower in 'case_f3n2e100', #EPSG:4326. 9000x9000, 3:1
                
            distance_transform_bf: 
                super slow
    """
    
    assert isinstance(mar, ma.MaskedArray)
    assert mar.mask.any()
    assert not mar.mask.all()
    
    plog = lambda msg: print(msg) if log is None else log.debug(msg)
    
    #retrieve func
    plog(f'_distance_fill w/ \'{method}\' {kwargs}')
    f = getattr(scipy.ndimage, method)
    
    indices_ar = f(
        mar.mask.astype(int), 
        return_indices=True, 
        return_distances=False,
        **kwargs)
    
    #use the computed indicies to map the source values onto the target
    filled_ar = mar.data.copy() 
    filled_ar[mar.mask] = mar.data[tuple(indices_ar[:, mar.mask])]
    

    plog(f'finished _distance_fill')
    return filled_ar




def downscale_costGrow_xr(dem_fine_xr, wse_coarse_xr,
                          
                distance_fill='neutral',
                          
                 logger=None,
                 write_meta=True,
                 debug=__debug__,
                 out_dir=None,
                 ):
    """
    downscale a coarse WSE grid using costGrow algos and xarray
    
    params
    --------
    distance_fill: str
        type of cost surface to use
            neutral: nn flat surface extrapolation
            terrain_penalty: cost distance extrapolation with terrain penalty
            
            
    out_dir: str, optional
        output directory for debugging intermediaries
    """
    
    #=======================================================================
    # defaults
    #=======================================================================
    log = logger.getChild('costGrow')
    
    meta_d=dict()
    nodata = dem_fine_xr.rio.nodata
    dem_mask = dem_fine_xr.to_masked_array().mask
    #===========================================================================
    # pre-chescks
    #===========================================================================
    phaseName = '00_inputs'
    log.debug(phaseName)
    
    if debug:
        assert_dem_xr(dem_fine_xr, msg='DEM')
        assert_wse_xr(wse_coarse_xr, msg='WSE')        
        assert_equal_extents_xr(dem_fine_xr, wse_coarse_xr, msg='\nraw inputs')
        
        out_dir = _get_od(out_dir)
            
                            
        log.debug(f'out_dir set to \n    {out_dir}')
        

        
    #===========================================================================
    # function helpers------
    #===========================================================================
    if debug:
        def to_gtiff(da, phaseName, layerName=None):
            
            #output path
            shape_str = _get_zero_padded_shape(da.shape)
            if layerName is None:
                layerName = da.attrs['layerName']
 
            ofp = os.path.join(out_dir, f'{phaseName}_{layerName}_{shape_str}')
                
            assert not os.path.exists(ofp), ofp
                
            #write GeoTiff (for debugging)
            meta_d[f'{phaseName}_{layerName}'] = xr_to_GeoTiff(da, ofp+'.tif', log=log, compress=None) 
            
            #write pickle (for unit tests) 
            _write_to_pick(da, os.path.join(out_dir, phaseName, layerName+'.pkl'))            
            #wrap
            
            
            return ofp
    
    
    #===========================================================================
    # setup
    #===========================================================================
    
    dem_fine_xr.attrs['layerName'] = 'dem_fine'
    wse_coarse_xr.attrs['layerName'] = 'wse_coarse'
    
    #dump inputs (needs to come after the debug)
    if debug:
        to_gtiff(wse_coarse_xr, phaseName)
        to_gtiff(dem_fine_xr, phaseName)
    
    #get rescaling value
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
             'dem_mask_cnt':dem_mask.sum(),
             'wse_wet_cnt':np.invert(wse_mask).sum(),
             'debug':debug,
             })
        
    log.info(f'passed all checks and downscale={downscale}\n    {meta_d}')
    
 
    
    #===========================================================================
    # 01 resample------
    #===========================================================================
    phaseName='01_resamp'
    log.debug(phaseName)
    
    wse_fine_xr1 = resample_match_xr(wse_coarse_xr, dem_fine_xr, resampling=Resampling.bilinear)
    wse_fine_xr1.attrs['layerName']='wse_fine'
    
    if write_meta:
        meta_d[f'{phaseName}_wseFine_wetCnt']=np.invert(wse_fine_xr1.to_masked_array().mask).sum()
        
    if debug:
        assert_wse_xr(wse_fine_xr1)
        assert_equal_raster_metadata(wse_fine_xr1, dem_fine_xr, msg=phaseName)
        
        to_gtiff(wse_fine_xr1, phaseName)
        

    #===========================================================================
    # 02 wet partials--------
    #===========================================================================
    phaseName='02_wp'
    log.debug(phaseName)
    
    wse_mar = wse_fine_xr1.to_masked_array()
    dem_mar = dem_fine_xr.to_masked_array()
    
    
    
    wse_mar2 = ma.MaskedArray(np.nan_to_num(wse_mar.data, nodata),
              mask=np.logical_or(
                  np.logical_or(dem_mar.mask, wse_mar.mask), #union of masks
                  wse_mar<=dem_mar, #below ground water
                  ))
    
    wse_fine_xr2 = dataarray_from_masked(wse_mar2, wse_fine_xr1)
    
    
    #===========================================================================
    # post
    #===========================================================================
    if write_meta:
        meta_d['wse_fine_2wp_wet_cnt']=np.invert(wse_fine_xr2.to_masked_array().mask).sum()
        
        
    if debug:
        if not wse_mar.mask.sum()<=wse_mar2.mask.sum():
            raise AssertionError('expected wet-cell count to decrease during wet-partial treatment')        
 
        to_gtiff(wse_fine_xr2, phaseName)
        
        assert_xr_geoTiff(wse_fine_xr2)
        
 
        
    #===========================================================================
    # 03 dry partials--------
    #===========================================================================
    phaseName='03_dp'
    log.debug(f'{phaseName} w/ distance_fill={distance_fill}')
 
    #===========================================================================
    # get imputed/filled WSE
    #===========================================================================
        
    if distance_fill == 'neutral':
        wse_filled_ar = _distance_fill(wse_fine_xr2.to_masked_array(), log=log.getChild(phaseName)) 
        
    elif distance_fill == 'terrain_penalty':         
        wse_filled_xr = _distance_fill_cost_terrain(wse_fine_xr2, dem_fine_xr,
                                    wse_coarse_xr, log=log.getChild(phaseName),
                                    out_dir=out_dir)
        
        wse_filled_ar = wse_filled_xr.data
 
    else:
        raise KeyError(distance_fill)
    
    #===========================================================================
    # #infill w/ valid wses
    #===========================================================================
    log.debug(f'infilling w/ valids')
    wse_ar = np.where(wse_filled_ar>dem_mar.data, wse_filled_ar, np.nan)
        
    
    wse_fine_xr3 = dataarray_from_masked(
        ma.MaskedArray(wse_ar, mask=np.logical_or(np.isnan(wse_ar), dem_mask)), wse_fine_xr2)
    
    
    #===========================================================================
    # post
    #===========================================================================
    if debug:
        if not wse_fine_xr3.isnull().sum()<=wse_fine_xr2.isnull().sum():
            raise AssertionError(f'dry-cell failed to decrease during {phaseName}')        
 
        to_gtiff(wse_fine_xr3, phaseName)
    
    return None, meta_d
 
 
    
    
        

    
    
 
