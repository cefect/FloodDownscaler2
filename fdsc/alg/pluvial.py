'''
Created on Jun. 2, 2024

@author: cef
'''


#===============================================================================
# imports----------
#===============================================================================
import os, argparse, logging, datetime, tempfile, pickle, copy, gc
import rioxarray
from rasterio.enums import Resampling
from tqdm.auto import tqdm

import scipy.ndimage 
import skimage.graph
import scipy

#import memory_profiler
 
#from osgeo import gdal

#from ...parameters import today_str
from ..hp.dirz import get_od
from ..hp.logr import get_new_file_logger, get_log_stream
from ..hp.xr import (
    resample_match_xr, dataarray_from_masked, xr_to_GeoTiff, approximate_resolution_meters,
    wse_to_wsh_xr, plot_histogram_with_stats
    )
from ..assertions import *

from ..coms import shape_ratio, set_da_layerNames


#===============================================================================
# HELPESR
#===============================================================================
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

def _wet(da):
    return float(da.isnull().sum().item()/da.size)

def _to_gtiff(da, phaseName, out_dir=None, meta_d=dict(), layerName=None, log=None):
    
    assert_xr_geoTiff(da, msg=f'output to_gtiff, {phaseName}.{layerName}')
    
    #output path
    shape_str = _get_zero_padded_shape(da.shape)
    if layerName is None:
        layerName = da.attrs['layerName']

    ofp = os.path.join(out_dir, f'{phaseName}_{layerName}_{shape_str}')                
    assert not os.path.exists(ofp), ofp
        
    #write GeoTiff (for debugging)            
    meta_d[f'{phaseName}_{layerName}'] = xr_to_GeoTiff(da, ofp+'.tif', log=log, 
                                                       compress=None) 
    
    #write pickle (for unit tests) 
    #_write_to_pick(da, os.path.join(out_dir, phaseName, layerName+'.pkl'))            
    #wrap
    
    
    return ofp + '.tif'

#===============================================================================
# RUNNERS------
#===============================================================================
   



def filter_small_groups(wet_bar, small_pixel_count, log):
    """
    
    params
    ----------
    dry_bar: np.array boolean
        True=Dry, False=Wet
    
    returns
    ----------
    cells part of small groups
    """
    assert wet_bar.any()
    
    inun_fine_ar = np.where(wet_bar, 1.0, 0.0) #0:dry, 1:wet
    # produce integer labels for each connected component
    #0-valued pixels are considered as background pixels
    labels, nlabels = skimage.measure.label(
        inun_fine_ar, 
        connectivity=1, 
        return_num=True)
    
    
    
    log.debug(f'identified {nlabels} groups')
    #get the size of each region
    props = skimage.measure.regionprops_table(labels, properties=('label', 'area'))
    ser = pd.Series(dict(zip(props['label'], props['area']))).astype(int).sort_values().rename('pixel_cnt')
    #identify regions below the threshold
    small_labels_ar = ser[ser <= small_pixel_count].index.values
    
    if len(small_labels_ar)==nlabels:
        raise AssertionError(f'all {nlabels} groups were filtered by small_pixel_count={small_pixel_count}')
    
    if len(small_labels_ar)>0:
        log.debug(f'found {len(small_labels_ar)}/{nlabels} regions smaller than {small_pixel_count} pixels')
        #mask these out
        small_bar = np.isin(labels, small_labels_ar)
        
        assert small_bar.any()
     
        log.debug(f'filtering {small_bar.sum()} pixels falling in small regions')
    else:
        log.debug(f'no groups < {small_pixel_count} found')
        
        small_bar = np.full(wet_bar.shape, False)
        
 
    return small_bar

def _pluvial_pre(dem_fine_xr, wse_coarse_xr, filter_method, filter_depth, filter_depth_mode_buffer, 
                 small_pixel_count, dem_coarse_xr, dem_coarse_resampling, wsh_coarse_xr, 
                 
                 debug=__debug__, 
                 write_meta=False, 
                 out_dir=None, meta_d=dict(), wet_d=dict(), log=None
                 ):
    
    #===========================================================================
    # defaults
    #===========================================================================
    nodata = dem_fine_xr.rio.nodata
    wse_coarse_mar = wse_coarse_xr.to_masked_array()
 
    #===========================================================================
    # helpers-------
    #===========================================================================
    to_gtiff = lambda da, phaseName, layerName=None:_to_gtiff(da, phaseName, 
                  layerName=layerName, meta_d=meta_d, log=log, out_dir=out_dir)
    
    def upd_wet(da, k=None):
        if k is None: k = da.attrs['layerName']
        wet_d[k] = _wet(da)
    #===========================================================================
    # pre01 coarse WSH------
    #===========================================================================
    """neded by all filter methods"""
    phaseName = '00_01_WSHcoarse'
    check_against_dem = False
    if wsh_coarse_xr is None:
        log.debug(f'{phaseName} extrapolating coarse WSH from WSE')
        if dem_coarse_xr is None:
            #get a lower-bound DEM to ensure we are never above teh WSE
            log.warning(
                f'constructing coarse DEM from fine w/ {dem_coarse_resampling}' + '\nthis can lead to poor results in steep terrain' + '\nconsider passing the coarse WSH or coarse DEM explicitly' + '\nat a minimum check the resulting coarse_WSH and change the \'dem_coarse_resampling\' parameter if necessary')
            """no good way of inferring the correct 'dem_coarse_resampling' parameter
    
        needs to match the inferred relation between the coarse and the fine
    
        best is for the user to pass the coarse layers excplitily
    
        
    
        """
            dem_coarse_xr = resample_match_xr(dem_fine_xr, wse_coarse_xr, resampling=dem_coarse_resampling, 
                debug=debug)
        else:
            check_against_dem = True
        #get the coarse WSH
        wsh_coarse_xr = wse_to_wsh_xr(dem_coarse_xr, wse_coarse_xr, log=log, 
            assert_partial=False, allow_low_wse=True, debug=debug)
    else:
        log.debug(f'using user-supplied coarse WSH')
        
    #extract some wsh
    wsh_coarse_mar = wsh_coarse_xr.to_masked_array()
    domain_coarse_mask = copy.deepcopy(wsh_coarse_mar.mask)
    
    #check mask
    if domain_coarse_mask.any():
        assert wse_coarse_mar.mask[domain_coarse_mask].all(), f'some WSE cells outside the domain are unmasked'
    else:
        log.debug(f'domain is unmasked')
    log.debug(f'finished WSH construction w/ {wsh_coarse_xr.shape} and {domain_coarse_mask.sum()}/{domain_coarse_mask.size} masked domain cells')
    
    if debug:
        to_gtiff(wsh_coarse_xr, phaseName)
        if not dem_coarse_xr is None:
            to_gtiff(dem_coarse_xr, phaseName)
        #plot for parameterization
        meta_d['wsh_coarse_histogram_fp'] = plot_histogram_with_stats(
            wsh_coarse_xr, log=log, out_dir=out_dir)
    if write_meta:
        upd_wet(wse_coarse_xr, phaseName)
    #===========================================================================
    # get the filter depth-------
    #===========================================================================
    if filter_depth is None:
        log.debug(f'no filter_depth provided. extracting from mode(WSH)')
        ar = wsh_coarse_xr.round(3).values.ravel()
        filtered_ar = ar[~np.isnan(ar)] # Remove NaN values
        mode_result = scipy.stats.mode(filtered_ar)
        filter_depth = mode_result[0] + filter_depth_mode_buffer
        log.debug(f' computed as {mode_result}')
        meta_d['filter_depth'] = filter_depth
    #===========================================================================
    # pre02 blanket filter-------
    #===========================================================================
    phaseName = '00_02_blanket'
    if filter_method == 'blanket':
        #construct the blanket
        blanket_coarse_xr = xr.apply_ufunc(np.minimum, wsh_coarse_xr, filter_depth
                                           ).where(wse_coarse_xr.notnull(), 0.0
                                                   ).where(~domain_coarse_mask, np.nan
                                                           ).rio.write_nodata(nodata)
        if debug:
            assert blanket_coarse_xr.max().item() <= (filter_depth + 1e-5) #floating point issue
        #check this is dry everywhere the wse is dry (inside the domain)
            assert np.all(blanket_coarse_xr.data[np.logical_and(wse_coarse_xr.isnull(), ~domain_coarse_mask)] == 0)
    else:
        raise KeyError(filter_method)
    #warp
    blanket_coarse_xr.attrs['layerName'] = 'blanket_coarse'
    log.debug(f'blanket built w/ filter_method = {filter_method} and max={blanket_coarse_xr.max().item()}')
    if debug:
        to_gtiff(blanket_coarse_xr, phaseName + '_blanket')
    #===========================================================================
    # pre03 apply the blanket--------
    #===========================================================================
    phaseName = '00_03_filter'
    #lower hte WSE values
    wse_coarse_xr1 = wse_coarse_xr - blanket_coarse_xr
    #detect dry cells
    """could also get this by comparing against the coarse DEM, but this adds a dependency
    
    might need to move this dry detection inside the filter_method?"""
    dry_bx = wsh_coarse_xr.data <= filter_depth
    assert dry_bx.any(), f'filter_depth {filter_depth} failed to filter any pixels'
    wse_coarse_xr1 = wse_coarse_xr1.where(~dry_bx, np.nan)
    #add some metadata
    wse_coarse_xr1.attrs = wse_coarse_xr.attrs.copy()
    wse_coarse_xr1.attrs.update({'filter_method':filter_method, 'filter_depth':filter_depth})
    log.debug(f'filtered WSE_coarse from {_wet(wse_coarse_xr):.2%} to {_wet(wse_coarse_xr1):.2%} dry cells')
    #check that this is higher than the DEM everywhere
    if debug:
        to_gtiff(wse_coarse_xr1, phaseName)
        if check_against_dem:
            assert np.all(wse_coarse_xr1 >= dem_coarse_xr), f'got some negative water depthsa fter blanketing'
        if not dem_coarse_xr is None:
    #write the depths as well
            to_gtiff(wse_to_wsh_xr(dem_coarse_xr, wse_coarse_xr1, 
                    log=log, assert_partial=False, allow_low_wse=True, debug=False), phaseName)
        assert wse_coarse_xr1.isnull().sum() >= wse_coarse_xr.isnull().sum(), 'dry cells increased'
        assert np.all(np.isnan(wse_coarse_xr1.data[wse_coarse_xr.isnull()].ravel())), 'some dry raws are not dry anymore'
    if write_meta:
        upd_wet(wse_coarse_xr1, phaseName)
    #===========================================================================
    # pre04 remove small groups------------
    #===========================================================================
    phaseName = '00_03_smalls'
    if small_pixel_count > 0:
        wse_mar = wse_coarse_xr1.to_masked_array()
        raise IOError('check this')
        small_bar = filter_small_groups(~wse_mar.mask, small_pixel_count, log)
        wse_coarse_xr2 = wse_coarse_xr1.where(~small_bar, np.nan)
    else:
        wse_coarse_xr2 = wse_coarse_xr1
        small_bar = None
    if debug:
        to_gtiff(wse_coarse_xr2, phaseName)
        assert not wse_coarse_xr2.isnull().all()
        assert wse_coarse_xr2.isnull().sum() >= wse_coarse_xr1.isnull().sum()
    if write_meta:
        upd_wet(wse_coarse_xr2, phaseName)
        
    #===========================================================================
    # write some to disc
    #===========================================================================
        
    return wse_coarse_xr2, small_bar, wse_coarse_xr1,  blanket_coarse_xr, filter_depth

#@memory_profiler.profile 
def downscale_pluvial_xr(
        dem_fine_xr, wse_coarse_xr,
        
        wse_proj_func = None,
        filter_method='blanket',
        filter_depth=0.1,
        filter_depth_mode_buffer=0.01, 
        small_pixel_count=5,
        reapply_small_groups=False,
        
        
        dem_coarse_xr=None, dem_coarse_resampling=Resampling.average,
        wsh_coarse_xr=None,
        
        logger=None,
        #write_meta=True,
        debug=__debug__, write_meta=True,
        out_dir=None,  meta_d=dict(),
        
        **kwargs):
    """wraper for pluvial pre/post filtering
    
    because pluvial floods generally have small depths everywhere
        it is not viable to project out water surfaces
        therefore, we apply some pre/post processing before the WSE projection
        
    two methods for this pre-filter are implemented as described in 'filter_method'
        both rely on a coarse WSH
        
 
    
    three options for the coarse WSH are implemented (in order of preference):    
        provide the coarse WSH (wsh_coarse_xr not None) 
        provide the coarse DEM (wsh_coarse_xr is None and dem_coarse_xr not None)
        extrapolate coarse depths from coarse WSE
 
    
    
    Params
    ---------------
    filter_method: str, default 'blanket'
        how to prefilter the WSE prior to extrapolation
        
        'blanket':a fixed value (filter_depth) subtracted before extrapolation and re-applied
            
        'threshold':values below the treshold are removed and re-applied
            
    
    filter_depth: flaot
        depth used for pluvial filtering. if None, computed from the mode + filter_depth_mode_buffer
        
    filter_depth_mode_buffer: float
        when filter_depth=None (computed from mode), this buffer willb e applied
        
    small_pixel_count: int
        count of pixels for groups to remove then re-apply
        
    reapply_small_groups: bool
        whether to add back the small groups removed by small_pixel_count post wse extrapolation
    """
    
 
    
    #=======================================================================
    # defaults
    #=======================================================================
    log = logger.getChild('pluvial')    
 
 
    
    assert not wse_proj_func is None 
    nodata = wse_coarse_xr.rio.nodata
    
    if reapply_small_groups: assert small_pixel_count>0 
    #===========================================================================
    # setup
    #===========================================================================
    wse_coarse_mar = wse_coarse_xr.to_masked_array()
    wet_d = dict()
    meta_d=dict()

        
    if write_meta or debug:
        write_meta=True
        meta_d.update(dict(filter_depth=filter_depth, filter_method=filter_method))
        
        def upd_wet(da, k=None):
            if k is None: k = da.attrs['layerName']
            wet_d[k] = _wet(da)
            
        
    
    if debug:
        to_gtiff = lambda da, phaseName, layerName=None:_to_gtiff(da, phaseName, 
                  layerName=layerName, meta_d=meta_d, log=log, out_dir=out_dir)
 

        
 
    #===========================================================================
    # pre-process
    #===========================================================================
    log.info(f'starting pluvial pre-processing')
    wse_coarse_xr2, small_bar, wse_coarse_xr1,  blanket_coarse_xr, filter_depth= _pluvial_pre(dem_fine_xr, wse_coarse_xr, filter_method, filter_depth,                                  
                                  filter_depth_mode_buffer, small_pixel_count, dem_coarse_xr,                                  
                                  dem_coarse_resampling, wsh_coarse_xr, 
                                  debug=debug, write_meta=write_meta,                                  
                                  out_dir=out_dir, meta_d=meta_d, wet_d=wet_d, log=log)
        
    gc.collect()
    #===========================================================================
    # WSE extrapolation--------
    #===========================================================================
    log.info(f'executing downscaler w/ filtered WSE')
    wse_fine_xr_extrap, meta_d =  wse_proj_func(
        dem_fine_xr, wse_coarse_xr2,
        logger=logger, debug=debug, out_dir=out_dir,meta_d=meta_d,
        **kwargs)
    log.info('extrpolation finished... post-processing for fluvial')
    
    gc.collect()
    #===========================================================================
    # post01 reapplly small groups-------
    #===========================================================================
    phaseName='99_01_reapplySmalls'
    
    if (not small_bar is None) and reapply_small_groups:
        """not sure about this... is the wet partial working?"""
        log.debug('reapplying small groups')
        #resample the small guys        
        wse_fine_smalls_xr = resample_match_xr(wse_coarse_xr1.where(small_bar, np.nan),wse_fine_xr_extrap,
                          resampling=Resampling.nearest,debug=debug)
        
        #WP filter on dem
        #wse_fine_smalls_xr[0,0]=500
        wse_fine_smalls_wp_xr = wse_fine_smalls_xr.where(wse_fine_smalls_xr.fillna(dem_fine_xr)>dem_fine_xr, np.nan)
        wse_fine_smalls_wp_xr.attrs['layerName'] = 'wse_fine_smalls_wp'
        
        #infill 
        wse_fine_xr01 = wse_fine_xr_extrap.fillna(wse_fine_smalls_wp_xr)
        
        
        if debug:
            to_gtiff(wse_fine_xr01, phaseName)
            to_gtiff(wse_fine_smalls_wp_xr, phaseName)
            assert wse_fine_xr01.isnull().sum()<=wse_fine_xr_extrap.isnull().sum()
        
    else:
        wse_fine_xr01 = wse_fine_xr_extrap
        

    
 
    
    #===========================================================================
    # post02 reapply blanket----------
    #===========================================================================
    phaseName='99_02_removeBlanket'
    
    #add blanket to wet cells
    
    if filter_method=='blanket':
        
        blanket_fine_xr = resample_match_xr(blanket_coarse_xr,
                                wse_fine_xr01,resampling=Resampling.bilinear,debug=debug)
    
        #anywhere blanket has values, use the WSE (DEM where dry) + blanket
        wse_fine_xr1 = wse_fine_xr01.where(blanket_fine_xr.isnull(),  
                 wse_fine_xr01.fillna(dem_fine_xr) + blanket_fine_xr)
        
        if debug:
            to_gtiff(blanket_fine_xr, phaseName)
            assert np.all(blanket_fine_xr<=filter_depth+1e-5)
            assert np.all(blanket_fine_xr>=0.0)
        
    else:
        raise KeyError(filter_method)
    
    
    del wse_fine_xr_extrap
    
    if debug:
 
        to_gtiff(wse_fine_xr1, phaseName)
        assert np.all(wse_fine_xr1>dem_fine_xr)
        
        #write the depths as well
        to_gtiff(wse_to_wsh_xr(dem_fine_xr, wse_fine_xr1, log=log, assert_partial=False, debug=False),phaseName)
        
    if write_meta:
        upd_wet(wse_fine_xr1, phaseName)
    
    
 
            
    #=======================================================================
    # wrap
    #=======================================================================
    if write_meta:
        meta_d.update({f'wetCnt_{k}':v for k,v in wet_d.items()})
    
    log.debug(f'finished pluvial wrapper')
    
    return wse_fine_xr1, meta_d
        
        
        
     