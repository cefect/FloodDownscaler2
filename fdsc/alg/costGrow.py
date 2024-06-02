'''
Created on May 25, 2024

@author: cef
'''

#===============================================================================
# imports----------
#===============================================================================
import os, argparse, logging, datetime, tempfile, pickle, copy
import rioxarray
from rasterio.enums import Resampling
from tqdm.auto import tqdm

import scipy.ndimage 
import skimage.graph
from osgeo import gdal, gdal_array

from parameters import today_str
from ..hp.dirz import get_od
from ..hp.logr import get_new_file_logger, get_log_stream
from ..hp.xr import (
    resample_match_xr, dataarray_from_masked, xr_to_GeoTiff, approximate_resolution_meters,
    wse_to_wsh_xr
    )
from ..assertions import *

from ..coms import shape_ratio, set_da_layerNames


import geopandas as gpd
from shapely.geometry import Point, LineString
import pyproj

#===============================================================================
# HELPERS------
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


def _distance_fill_cost_terrain(wse_fine_xr, dem_fine_xr, wse_coarse_xr, log=None,
                                cd_backend='wbt',out_dir=None,m_to_rad=None, 
                                **kwargs
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
 
    log.debug(f'computing cost surface')
    cost_xr = xr.where(delta_xr < 0, 0.0, delta_xr).fillna(999).round(1)
    cost_xr.rio.write_crs(dem_fine_xr.rio.crs, inplace=True) #not sure where this got lost
    
    if m_to_rad:
        raise IOError('dome')
        cost_xr
    
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
                        out_dir = os.path.join(get_od(out_dir), 'wbt'), **kwargs)
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

    current_wdir = os.getcwd() #WBT moves htis
    restore_cwd = lambda: os.chdir(current_wdir)
    out_dir = get_od(out_dir)
    
    to_gtif = lambda da, fn: xr_to_GeoTiff(da, os.path.join(out_dir, fn), log)
    
    
    #===========================================================================
    # init wbt
    #===========================================================================
    from ..hp.wbt import wbt
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
    
    restore_cwd()
    assert current_wdir==os.getcwd()
 
    #=======================================================================
    # costAllocation
    #=======================================================================
    log.debug("wbt.cost_allocation")
    costAlloc_fp = os.path.join(out_dir, '02_costAllocation.tif')
    if not wbt.cost_allocation(wse_fp, backlink_fp, costAlloc_fp) == 0:
        raise IOError('wbt.cost_allocation')
    
    restore_cwd()
    log.debug(f'wrote to \n    {costAlloc_fp}')
    #===========================================================================
    # back to Xarray
    #===========================================================================
    load_xr = lambda x: rioxarray.open_rasterio(x,masked=False).squeeze().rio.write_nodata(-9999)
    
    wse_filled_xr = load_xr(costAlloc_fp) - add_increment
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




def _meters_to_latlon(distance_meters, latitude, longitude, crs="EPSG:4326"):
    """Converts a distance in meters to a latitude and longitude difference at a given location.

    Args:
        distance_meters: The distance in meters.
        latitude: The latitude of the starting point (in degrees).
        longitude: The longitude of the starting point (in degrees).
        crs: The coordinate reference system of the input coordinates (default WGS84).

    Returns:
        A tuple containing the latitude difference and longitude difference in degrees.
    """

    # Create a GeoDataFrame with the starting point
    point = Point(longitude, latitude)
    gdf = gpd.GeoDataFrame(geometry=[point], crs=crs)

    # Define a custom Azimuthal Equidistant projection centered at the point
    aeqd_crs = pyproj.CRS.from_proj4(
        f"+proj=aeqd +lat_0={latitude} +lon_0={longitude} +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
    )

    # Project to AEQD and buffer
    gdf_aeqd = gdf.to_crs(aeqd_crs)
    gdf_aeqd["geometry"] = gdf_aeqd.buffer(distance_meters)

    # Project back to WGS84 and get the new coordinates
    gdf_wgs84 = gdf_aeqd.to_crs(crs)
    new_coords = gdf_wgs84.geometry[0].centroid  

    # Calculate the differences
    lat_diff = new_coords.y - latitude
    lon_diff = new_coords.x - longitude

    return lat_diff, lon_diff


def _xr_gdalslope(dataarray, log=None, 
                    alg="Horn", # literature suggests Zevenbergen & Thorne to be more suited to smooth landscapes, whereas Horn's formula to perform better on rougher terrain.
                    slopeFormat="percent",
                    scale=1,
                    computeEdges=True,
                    #zFactor=1,
        
                  **kwargs):
    """compute teh slope of a rioxarray by sending it to gdal"""
    
 
    
    # Create in-memory source raster
    drv = gdal.GetDriverByName("MEM")  
    src_ds = drv.Create("", dataarray.shape[1], dataarray.shape[0], 1, gdal.GDT_Float32)
    src_band = src_ds.GetRasterBand(1)
    src_band.WriteArray(dataarray.data)
    src_band.SetNoDataValue(int(dataarray.rio.nodata) or -9999)  # Set nodata from DataArray
 
    # Set GeoTransform and Projection
    # Extract GeoTransform from Affine object
    #===========================================================================
    # transform = dataarray.rio.transform()
    # geotransform = (transform.a, transform.b, transform.c, 
    #                 transform.d, transform.e, transform.f)
    #===========================================================================
     
    #src_ds.SetGeoTransform(geotransform)
    src_ds.SetProjection(dataarray.rio.crs.to_wkt())

    # Create in-memory destination raster
    #===========================================================================
    # dst_ds = drv.Create("", dataarray.shape[1], dataarray.shape[0], 1, gdal.GDT_Float32) 
    # dst_ds.SetGeoTransform(src_ds.GetGeoTransform())
    # dst_ds.SetProjection(src_ds.GetProjection())
    #===========================================================================
    #===========================================================================
    # dataarray.rio.__gdal_obj__
    # dataarray.rio.to_rasterio()
    # with dataarray.rio.rasterio as src:
    #     src_ds = src.ds  # Get the GDAL dataset object
    #===========================================================================


    #src_ds = dataarray.rio.open_gdal_dataset()
    # Perform GDAL DEM processing
    
    dst_ds = gdal.DEMProcessing(
        "",
        src_ds,
        "slope",
        format="MEM", 
        alg=alg, slopeFormat=slopeFormat, scale=scale, computeEdges=computeEdges  
    )
    
 
    dst_band = dst_ds.GetRasterBand(1)
 
    # Create xarray DataArray from result
    ar = copy.deepcopy(dst_band.ReadAsArray())
    
    
    mask = np.logical_or(ar==dataarray.rio.nodata, dataarray.isnull().data)
    mar = ma.MaskedArray(ar, mask=mask)
    
    da = dataarray_from_masked(mar, dataarray)
    
    da.attrs['layerName'] = 'slope'
 
    return da
    


def _cost_accumulation_wbt(wse_xr, cost_xr, log=None, out_dir=None):
    """
    Fills masked values in a 2D array using cost-distance weighted nearest neighbor interpolation.
    """
    
    #===========================================================================
    # helpers for wbt
    #===========================================================================
    current_wdir = os.getcwd() #WBT moves htis
    restore_cwd = lambda: os.chdir(current_wdir)
    out_dir = get_od(out_dir)
    
    to_gtif = lambda da, fn: xr_to_GeoTiff(da, os.path.join(out_dir, fn), log)
    
    
    #===========================================================================
    # init wbt
    #===========================================================================
    from ..hp.wbt import wbt
    wbt.set_default_callback(lambda value: log.debug(value) if not "%" in value else None)
            
    #===========================================================================
    # prep
    #===========================================================================
    
 #==============================================================================
 #    #add arbitrary increment to handle negative WSE
 #    add_increment=0
 # 
 #    if wse_xr.min()<0:
 #        """shouldnt need to revert because its only teh costs we care about"""
 #        add_increment=9999
 #        log.warning(f'adding increment {add_increment} to handle negative WSE')    
 #        wse_xr = wse_xr + add_increment
 #==============================================================================
    #convert to inundation
    inun_xr = wse_xr.where(wse_xr.notnull(), 0.0).where(wse_xr.isnull(), 1.0)
    #inun_xr = np.where(wse_xr.notnull(), 1.0, 0.0)
 
    assert inun_xr.sum()==wse_xr.notnull().sum()
    
    #dump to GeoTiff
    log.debug(f'dumping Xarrays to GeoTiffs')
    inun_fp = to_gtif(inun_xr, '00_inun.tif')
    cost_fp = to_gtif(cost_xr, '00_cost.tif')
    
    
    #===========================================================================
    # #compute backlink raster
    #===========================================================================
    log.debug("wbt.cost_distance")
    backlink_fp = os.path.join(out_dir, f'01_backlink.tif')
    costAccum_fp = os.path.join(out_dir, f'01_outAccum.tif')
    if not wbt.cost_distance(inun_fp,cost_fp, 
                             costAccum_fp,backlink_fp,
                             ) == 0:
        raise IOError('cost_distance')
    
    restore_cwd()
    assert current_wdir==os.getcwd()
    
    #===========================================================================
    # back to Xarray
    #===========================================================================
    load_xr = lambda x: rioxarray.open_rasterio(x,masked=False).squeeze(
        ).compute().rio.write_nodata(-9999).rio.write_crs(wse_xr.rio.crs)
    
    costAccum_xr = load_xr(costAccum_fp)
    costAccum_xr.attrs = wse_xr.attrs.copy()
    
    log.debug('finished')
    
    return costAccum_xr

#===============================================================================
# RUNNERS------
#===============================================================================
   
            
            




def downscale_costGrow_xr(dem_fine_xr, wse_coarse_xr,
                          
                distance_fill='neutral',
                distance_fill_method='distance_transform_cdt',
                
                decay_method='linear',
                decay_frac=0.001,
                dp_coarse_pixel_max=10,
                          
                 logger=None,
                 write_meta=True, meta_d=dict(),
                 debug=__debug__,
                 out_dir=None,
                 ):
    """
    downscale a coarse WSE grid using costGrow algos and xarray
    
    params
    --------
    distance_fill: str, default 'neutral'
        type of cost surface to use
            neutral: nn flat surface extrapolation
            terrain_penalty: cost distance extrapolation with terrain penalty
            
    
    distance_fill_method: str, default 'distance_fill_method'
        scipy.ndimage method to apply. see _distance_fill()
        
        
            
    
    decay_frac: float, default 0.001 m/m
        value (in meters) multiiplied  by distance and pixel size to obtain the decay grid
        decay grid is subtracted from Dry-Partial WSE
        equivalent to the water slope applied to the Dry-Partial growth
        pass 0.0 or None to skip applying decay
        
    dp_coarse_pixel_max: int, default 10.0
        maximum number of coarse pixels to allow dry-partial growth
        pass None to skip applying this threshold (unbounded growth)
         
        
            
    out_dir: str, optional
        output directory for debugging intermediaries
    """
    
    #=======================================================================
    # defaults
    #=======================================================================
    log = logger.getChild('costGrow')
    
 
    nodata = dem_fine_xr.rio.nodata
    dem_mar = dem_fine_xr.to_masked_array()
    dem_mask = dem_mar.mask
    
    if not dem_fine_xr.rio.crs.linear_units == 'metre':
        res_t = approximate_resolution_meters(dem_fine_xr)        
        m_to_rad = np.abs(res_t).mean()/np.abs(dem_fine_xr.rio.resolution()).mean()
        
    else:
        res_t = dem_fine_xr.rio.resolution()
        m_to_rad=None
    #===========================================================================
    # pre-chescks
    #===========================================================================
    phaseName = '00_inputs'
    log.debug(phaseName)
    
    if decay_frac==0.0: assert decay_method=='none'
        

        
    #===========================================================================
    # function helpers------
    #===========================================================================
    if debug:
        out_dir = get_od(out_dir)
        log.debug(f'out_dir set to \n    {out_dir}')
        
        def to_gtiff(da, phaseName, layerName=None):
            
            assert_xr_geoTiff(da, msg=f'output to_gtiff, {phaseName}.{layerName}')
            
            #output path
            shape_str = _get_zero_padded_shape(da.shape)
            if layerName is None:
                layerName = da.attrs['layerName']
 
            ofp = os.path.join(out_dir, f'{phaseName}_{layerName}_{shape_str}')
                
            assert not os.path.exists(ofp), ofp
                
            #write GeoTiff (for debugging)
            
            meta_d[f'{phaseName}_{layerName}'] = xr_to_GeoTiff(da, ofp+'.tif', log=log, compress=None) 
            
            #write pickle (for unit tests) 
            #_write_to_pick(da, os.path.join(out_dir, phaseName, layerName+'.pkl'))            
            #wrap
            
            
            return ofp + '.tif'
        
        
 
    
    #===========================================================================
    # setup
    #===========================================================================
    
    # Iterate through the DataArrays and check their layerName attributes
    set_da_layerNames({'dem_fine':dem_fine_xr,'wse_coarse':wse_coarse_xr})
        
    
    #===========================================================================
    # dem_fine_xr.attrs['layerName'] = 'dem_fine'
    # wse_coarse_xr.attrs['layerName'] = 'wse_coarse'
    #===========================================================================
    
    #dump inputs (needs to come after the debug)
    if debug:
        to_gtiff(wse_coarse_xr, phaseName)
        to_gtiff(dem_fine_xr, phaseName)
        assert_dem_xr(dem_fine_xr, msg='DEM')
        assert_wse_xr(wse_coarse_xr, msg='WSE')        
        assert_equal_extents_xr(dem_fine_xr, wse_coarse_xr, msg='\nraw inputs')    
    
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
             'wse_coarse_wet_cnt':np.invert(wse_mask).sum(),
             'debug':debug,
             })
        
        wet_d = dict()
        def upd_wet(da, k):
            wet_d[k] = da.notnull().sum().item()
            
        
        
    log.info(f'passed all checks and downscale={downscale}\n    {meta_d}')
    
 
    
    #===========================================================================
    # 01 resample------
    #===========================================================================
    phaseName='01_resamp'
    log.debug(phaseName)
    
    wse_fine_xr1 = resample_match_xr(wse_coarse_xr, dem_fine_xr, resampling=Resampling.bilinear)
    wse_fine_xr1.attrs['layerName']='wse_fine'
    
    if write_meta:
        upd_wet(wse_fine_xr1, phaseName)
         
        
    if debug:
        ofp = to_gtiff(wse_fine_xr1, phaseName)
        assert_wse_xr(wse_fine_xr1, msg=f'{phaseName}\n{ofp}')
        assert_equal_raster_metadata(wse_fine_xr1, dem_fine_xr, msg=f'{phaseName}\n{ofp}')
        
        
        

    #===========================================================================
    # 02 wet partials--------
    #===========================================================================
    phaseName='02_wp'
    log.debug(phaseName)
    
    wse_mar = wse_fine_xr1.to_masked_array()
 
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
        upd_wet(wse_fine_xr2, phaseName)
        
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
    
    
    #needed by filter ops
    f = getattr(scipy.ndimage, distance_fill_method)
    distance_ar = f(wse_fine_xr2.isnull().data.astype(int),return_indices=False,return_distances=True)
    
    #===========================================================================
    #    03.1 growth threshold------
    #===========================================================================
    if not dp_coarse_pixel_max is None:
        #True=within region of interest
        grow_thresh_bar = distance_ar/downscale<dp_coarse_pixel_max
        log.debug(f'w/ dp_coarse_pixel_max={dp_coarse_pixel_max} masked {np.invert(grow_thresh_bar).sum()/grow_thresh_bar.size:.4f} of pixels')
    else:
        grow_thresh_bar = np.full(distance_ar.shape, True)
        
        
 
    #===========================================================================
    #    03.2 get imputed/filled WSE---------
    #===========================================================================
    """seems like there should be  away to apply grow_thresh_bar to speed up the distance calcs below""" 
        
    if distance_fill == 'neutral':
        wse_filled_ar = _distance_fill(wse_fine_xr2.to_masked_array(), log=log.getChild(phaseName),
                                       method=distance_fill_method)
        
        if debug:
            wse_filled_xr = dataarray_from_masked(ma.MaskedArray(wse_filled_ar), wse_fine_xr2)
        
    elif distance_fill == 'terrain_penalty':         
        wse_filled_xr = _distance_fill_cost_terrain(wse_fine_xr2, dem_fine_xr,
                                    wse_coarse_xr, log=log.getChild(phaseName),
                                    out_dir=out_dir, m_to_rad=m_to_rad)
        
        wse_filled_ar = wse_filled_xr.data
 
    else:
        raise KeyError(distance_fill)
    

    #===========================================================================
    #    03.3 decay--------
    #===========================================================================
    
    if not decay_method=='none':
        
        log.debug(f'applying decay_method={decay_method} w/ decay_frac={decay_frac}')
        #convert the decay_frac to geographic
        decay_frac_m = np.abs(res_t).mean()*decay_frac            
        meta_d['decay_frac_m'] = decay_frac_m
    
    #===========================================================================
    # by method
    #===========================================================================
    if decay_method=='linear':
        
        #multiply by distance to nearest wet
        log.debug(f'applying distanc-weighted decay_frac_m={decay_frac_m:.3f}')        
        decay_ar = distance_ar*decay_frac_m
        
    elif decay_method=='slope_weighted':
        
        #retrieve teh slope
        slope_xr = _xr_gdalslope(dem_fine_xr, log=log, scale=np.abs(res_t).mean())
        
        #cost weight this starting from teh wet partials
        costAccum_slope_xr = _cost_accumulation_wbt(wse_fine_xr2, slope_xr, log=log, 
                                                    out_dir=os.path.join(out_dir, decay_method),
                                                    )
        
        #convert to meters
        if not m_to_rad is None:

            log.debug(f'converting distance degrees to meters w/ {m_to_rad:.2f}')
            costAccum_slope_xr = costAccum_slope_xr*m_to_rad
        
        #multiply  by decay fraction
        log.debug(f'computing decay from cost-accumulated slopes * {decay_frac_m:.3f}')
        decay_ar = costAccum_slope_xr.data * decay_frac_m
        
        
        
    else:
        decay_ar = np.zeros(distance_ar.shape)
    
    #===========================================================================
    # 03.4 infill w/ valid wses
    #===========================================================================
    log.debug(f'infilling w/ valids')
    wse_ar = np.where(
        np.logical_and(grow_thresh_bar,
                       (wse_filled_ar-decay_ar)>dem_mar.data
                       ), wse_filled_ar, np.nan)
        
    
    wse_fine_xr3 = dataarray_from_masked(
        ma.MaskedArray(wse_ar, mask=np.logical_or(np.isnan(wse_ar), dem_mask)), wse_fine_xr2)
    
    
    #===========================================================================
    # post
    #===========================================================================
    if debug:
        if not wse_fine_xr3.isnull().sum()<=wse_fine_xr2.isnull().sum():
            raise AssertionError(f'dry-cell failed to decrease during {phaseName}')        
        
        #write the phase result
        to_gtiff(wse_fine_xr3, phaseName)
        
        #write the wse grow/filled
        wse_filled_xr.attrs['layerName']='2wse_fill'
        to_gtiff(wse_filled_xr, phaseName)
        
        
        
        #write growth threshold
        if not dp_coarse_pixel_max is None:
            grow_xr = dataarray_from_masked(ma.MaskedArray(grow_thresh_bar.astype(float), mask=dem_fine_xr.isnull().data), dem_fine_xr)
            grow_xr.attrs['layerName'] = '1grow_thresh'
            to_gtiff(grow_xr, phaseName)
            
        
        #write decay
        if not decay_method=='none':
            decay_xr = dataarray_from_masked(ma.MaskedArray(decay_ar, mask=dem_fine_xr.isnull().data), dem_fine_xr)
            decay_xr.attrs['layerName'] = '3decay'
            to_gtiff(decay_xr, phaseName)
            to_gtiff(slope_xr, phaseName)
        
    if write_meta:
        meta_d['distance_fill'] = distance_fill
        upd_wet(wse_fine_xr3, phaseName)
        
    #===========================================================================
    # 04  isolated----------
    #===========================================================================
    phaseName='04_isol'
    log.debug(f'{phaseName}  ')
    
    wse_mar = wse_fine_xr3.to_masked_array()
    inun_fine_ar = np.where(wse_mar.mask, 0.0, 1.0) #0:dry, 1:wet
    
    # produce integer labels for each connected component
    #0-valued pixels are considered as background pixels
    labels, nlabels = skimage.measure.label(
        inun_fine_ar, 
        connectivity=1, 
        return_num=True)
    
    log.debug(f'identified {nlabels} regions')
    #===================================================================
    # #identify those labels we want to keep
    #===================================================================
    #wet partial inundation
    inun_wp_bar = np.invert(wse_fine_xr2.to_masked_array().mask) #True=wet
 
    #those wet in 02WP
    connected_labels = np.unique(labels[inun_wp_bar])
    log.debug(f'{len(connected_labels)}/{nlabels} intersect w/ 02wp')
    
    #apply mask
    iso_bar = np.isin(labels, connected_labels)

    wse_fine_xr4 = wse_fine_xr3.where(iso_bar)
    
    #===========================================================================
    # post
    #===========================================================================
    if debug:
        if not wse_fine_xr4.isnull().sum()>=wse_fine_xr3.isnull().sum():
            raise AssertionError(f'dry-cells failed to decrease during {phaseName}')
        
        assert np.all(
            wse_fine_xr4.isnull().data.ravel()[dem_fine_xr.isnull().data.ravel()]), 'DEM nulls not set on result'
        
        #check all resampled wets are still wet
        if not np.logical_and(wse_fine_xr4.notnull(), wse_fine_xr2.notnull()).sum()==  wse_fine_xr2.notnull().sum():
            raise AssertionError(f'lost some wet cells from resample')
        
        assert_equal_raster_metadata(wse_fine_xr4, dem_fine_xr)
 
        to_gtiff(wse_fine_xr4, phaseName)
        
        #write depths also
        to_gtiff(wse_to_wsh_xr(dem_fine_xr, wse_fine_xr4, log=log, allow_low_wse=True), phaseName)
        
        
    if write_meta:
        meta_d['isolated_region_raw_cnt'] = nlabels
        meta_d['isolated_region_sel_cnt'] = len(connected_labels)
        upd_wet(wse_fine_xr4, phaseName)
        
    #===========================================================================
    # 05 WRAP------
    #===========================================================================
    #append wet counts to meta
    if write_meta:
        meta_d.update({f'wetCnt_{k}':v for k,v in wet_d.items()})
    
    log.debug(f'finished w/ {meta_d}')
    return wse_fine_xr4, meta_d
 
 
    
    
        

    
    
 
