'''
Created on May 25, 2024

@author: cef
'''
#===============================================================================
# IMPORTS---------
#===============================================================================
import os
import xarray as xr
import numpy as np
import numpy.ma as ma
from rasterio.warp import transform
from rasterio.enums import Resampling

from .dirz import get_od
from ..assertions import *


def plot_histogram_with_stats(da, bins=100, title=None, 
                              out_dir=None, ofp=None, 
                              log=None):
    """Plots a histogram of a DataArray with statistics in the lower right corner.

    Args:
        data_array (xr.DataArray): The xarray DataArray to plot.
        bins (int, optional): Number of histogram bins. Defaults to 100.
        title (str, optional): Title for the plot. Defaults to the DataArray's name.
    """
    import matplotlib.pyplot as plt
    from scipy import stats  # Import for mode calculation
    
    #===========================================================================
    # defaults
    #===========================================================================
    
    if 'layerName' in da.attrs:
        layerName= da.attrs['layerName']
    else:
        layerName = 'dataArray'

    # Extract and filter data
    ar = da.round(3).values.ravel()
    filtered_ar = ar[~np.isnan(ar)]  # Remove NaN values

    # Calculate statistics
    mean = np.mean(filtered_ar)
    min_val = np.min(filtered_ar)
    max_val = np.max(filtered_ar)
    mode = stats.mode(filtered_ar)  # Calculate mode
    nodata_count = np.sum(np.isnan(ar))

    # Create histogram
    plt.hist(filtered_ar, bins=bins, density=False, alpha=0.7, color='skyblue', edgecolor='black')
    plt.xlabel('Values')
    plt.ylabel('Frequency')
    plt.title(title or f'Histogram of {layerName}')
    plt.grid(axis='y', alpha=0.5)

    # Add text annotations
    text = f"Mean: {mean:.2f}\nMin: {min_val:.2f}\nMax: {max_val:.2f}\nMode: {mode}\nNoData Count: {nodata_count}"
    text+=f'\nshape:{da.shape}'
    text+=f'\nsize:{da.size}'
    text+=f'\nzero_cnt:{(ar==0).sum()}'
    plt.text(0.95, 0.05, text, transform=plt.gca().transAxes, ha='right', va='bottom')

    #plt.show()
    
    if ofp is None:
        out_dir = get_od(out_dir)
        ofp = os.path.join(out_dir, f'{layerName}_histogram_with_stats.svg')
    
    plt.savefig(ofp, dpi = 600, format = 'svg', transparent=True)
    
    log.info(f'wrote histogram to \n    {ofp}')
    
    return ofp

def xr_to_GeoTiff(da, raster_fp, log=None, compress='LZW'): 

    #directory setup
    if not os.path.exists(os.path.dirname(raster_fp)):
        os.makedirs(os.path.dirname(raster_fp))
    
    assert raster_fp.endswith('.tif')
    
    assert isinstance(da, xr.DataArray), type(da)
    
    
    #===========================================================================
    # prep
    #===========================================================================
    """force some additional confromance
    necessary as we are less strict with assert_xr_geoTiff than needed for conforming GeoTiffs"""
    #attrs = da.attrs.copy()
    
     
    da = da.assign_coords(band=1).expand_dims(dim='band')    
    da = da.fillna(-9999).rio.write_nodata(-9999) #.rio.write_crs(da.rio.crs)
    
    assert_xr_geoTiff(da, msg='precheck')
    #===========================================================================
    # write
    #===========================================================================
    
    da.rio.to_raster(raster_fp,  compute=True, compress=compress)
    
    #===========================================================================
    # post
    #===========================================================================
    
    msg = f'wrote xr.DataArray {da.shape} to \n    {raster_fp}'
    if not log is None:
        log.debug(msg)
    else:
        print(msg)
    
    return raster_fp

def resample_match_xr(da, target_da, resampling=Resampling.nearest,debug=__debug__,
                      **kwargs):
    """resampling with better treatment for masks"""
    
    
    #wse_coarse_xr.fillna(0.0).interp_like(dem_fine_xr, method='linear', assume_sorted=True, kwargs={'fill_value':'extrapolate'})
    nodata = da.rio.nodata
    resampled_da =  da.fillna(nodata).rio.reproject_match(
        target_da, nodata=nodata, resampling=resampling, **kwargs)
    
    #infill nan where there is nodata
    resampled_da = resampled_da.where(resampled_da!=nodata, np.nan)
    
    #basic nodata check
    if debug:
        if da.isnull().any():
            assert resampled_da.isnull().any()
        else:
            assert not resampled_da.isnull().any()
    
        
    return resampled_da



def dataarray_from_masked(masked_array, target_dataarray):
    """
    Creates an xarray DataArray from a masked array, filling masked values with the
    nodata value from a target DataArray. The new DataArray inherits coordinates
    and dimensions from the target DataArray.

    Args:
        masked_array (numpy.ma.MaskedArray): The masked array to convert.
        target_dataarray (xr.DataArray): The target DataArray to get nodata and metadata from.

    Returns:
        xr.DataArray: A new DataArray with filled masked values and the same
                      coordinates and dimensions as the target DataArray.
    """

    if not isinstance(masked_array, ma.MaskedArray):
        raise TypeError("Input must be a numpy masked array.")

    if not isinstance(target_dataarray, xr.DataArray):
        raise TypeError("Target must be an xarray DataArray.")
    
    assert 'float' in masked_array.data.dtype.name, f'got bad dtype: {masked_array.data.dtype.name}'

    # Get the nodata value from the target DataArray
    nodata = target_dataarray.rio.nodata

    # Replace masked values with nodata
    try:
        filled_array = ma.filled(masked_array, fill_value=np.nan)
    except Exception as e:
        raise IOError(e)

    # Create a new DataArray
    new_dataarray = xr.DataArray(
        filled_array,
        dims=target_dataarray.dims,
        coords=target_dataarray.coords,
        attrs=target_dataarray.attrs
    )

    # Ensure rio attributes are copied over
 
    return new_dataarray.rio.write_crs(target_dataarray.rio.crs).rio.write_nodata(nodata)



def approximate_resolution_meters(xds):
    """Approximates the resolution of an EPSG:4326 raster in meters.

    Args:
        xds: The rioxarray DataArray representing the raster.

    Returns:
        A tuple containing the approximate x and y resolution in meters.
    """
    
    # Earth's radius in meters
    EARTH_RADIUS = 6371000

    # Get latitude/longitude resolution (in degrees)
    x_res_degrees, y_res_degrees = xds.rio.resolution()
    
    if xds.rio.crs.linear_units == 'metre':
        return x_res_degrees, y_res_degrees
        
    if not xds.rio.crs.is_geographic:
        raise NotImplementedError(f'expect a geographic crs or a projected one in meters')
        

    # Get raster bounds
    bounds = xds.rio.bounds()
    min_lat, max_lat = bounds[1], bounds[3]
    avg_lat = (min_lat + max_lat) / 2

    # Convert latitude resolution to meters
    y_res_meters = y_res_degrees * (2 * np.pi * EARTH_RADIUS) / 360

    # Convert longitude resolution to meters (depends on latitude)
    x_res_meters = x_res_degrees * (2 * np.pi * EARTH_RADIUS * np.cos(np.radians(avg_lat))) / 360

    return x_res_meters, y_res_meters


#===============================================================================
# def approximate_distance_degrees_to_meters(xds):
#     """Approximates the resolution of an EPSG:4326 raster in meters.
# 
#     Args:
#         xds: The rioxarray DataArray representing the raster.
# 
#     Returns:
#         A tuple containing the approximate x and y resolution in meters.
#     """
#     
#     # Earth's radius in meters
#     EARTH_RADIUS = 6371000
# 
#     # Get latitude/longitude resolution (in degrees)
#     x_res_degrees, y_res_degrees = xds.rio.resolution()
#     
#     if xds.rio.crs.linear_units == 'metre':
#         return x_res_degrees, y_res_degrees
#         
#     if not xds.rio.crs.is_geographic:
#         raise NotImplementedError(f'expect a geographic crs or a projected one in meters')
#         
# 
#     # Get raster bounds
#     bounds = xds.rio.bounds()
#     min_lat, max_lat = bounds[1], bounds[3]
#     avg_lat = (min_lat + max_lat) / 2
# 
#     # Convert latitude resolution to meters
#     y_res_meters = y_res_degrees * (2 * np.pi * EARTH_RADIUS) / 360
# 
#     # Convert longitude resolution to meters (depends on latitude)
#     x_res_meters = x_res_degrees * (2 * np.pi * EARTH_RADIUS * np.cos(np.radians(avg_lat))) / 360
# 
#     return x_res_meters, y_res_meters
#===============================================================================


def get_center_latlon(xds):
    """Calculates the center latitude and longitude of a rioxarray DataArray.

    Args:
        xds: The rioxarray DataArray representing the raster.

    Returns:
        A tuple containing the latitude and longitude of the center in degrees.
    """

    bounds = xds.rio.bounds()
    center_x = (bounds[0] + bounds[2]) / 2
    center_y = (bounds[1] + bounds[3]) / 2

    # Check if CRS is already geographic (lat/lon)
    if xds.rio.crs.is_geographic:
        return center_y, center_x

    # Reproject center coordinates to WGS84 (EPSG:4326) if necessary
    dst_crs = "EPSG:4326"
    lon, lat = transform(xds.rio.crs, dst_crs, [center_x], [center_y])
    return lat[0], lon[0]


def wse_to_wsh_xr(dem_xr, wse_xr, log=None, assert_partial=True, allow_low_wse=False, debug=True):
    """convert a WSE to a WSH with the DEM
    
    
    Params
    -------------
    allow_low_wse: bool
        allow wse and dem grids where WSE<DEM to pass (other wise an exception is thrown)
    
    assert_partial: bool, default True
        check that the WSE grid is partially wet
    
    """
    plog = lambda msg: None if log is None else log.debug(msg)
    
    plog(f'wse_to_wsh_xr on {wse_xr.shape}')    
    
    wse_mar= wse_xr.to_masked_array()
    dem_mar = dem_xr.to_masked_array()
    
    #===========================================================================
    # precheck
    #===========================================================================
    if debug:
        msg = 'wse_to_wsh_xr precheck'
        assert_equal_raster_metadata(wse_xr, dem_xr, msg=msg)
        assert_wse_ar(wse_mar, msg=msg, assert_partial=assert_partial)
        assert_dem_ar(dem_mar, msg=msg)
        
    
    #delta_ar = np.nan_to_num(wse_mar.data - dem_mar.data, 0.0)
    delta_mar = ma.MaskedArray(
        wse_mar.filled(dem_mar.data).data - dem_mar.data, mask=dem_mar.mask
        )
        
    
    #assert_wse_vs_dem(wse_xr, dem_xr, msg='wse to wsh precheck', assert_partial=assert_partial)
    #===========================================================================
    # check if WSE values were below the DEM
    #===========================================================================
    bool_ar = delta_mar[~delta_mar.mask].ravel()>0.0
    if not bool_ar.all():
        msg = f'{bool_ar.sum():,}/{bool_ar.size:,} WSE pixels were at or below the DEM'
        if allow_low_wse:
            if not log is None:
                log.warning(msg)
            else:
                warnings.warn(msg)
        else:
            raise AssertionError(msg)
    
 
    #===========================================================================
    # build WSH
    #===========================================================================
    wsh_mar = ma.MaskedArray(np.where(delta_mar < 0.0, 0.0, delta_mar), 
        mask=dem_mar.mask, #mask where DEM is masked
        )
 
    wsh_xr = dataarray_from_masked(wsh_mar, dem_xr)
    
    if 'layerName' in wse_xr.attrs:
        wsh_xr.attrs['layerName'] = wse_xr.attrs['layerName'].lower().replace('wse', 'wsh')
        
    assert_wsh_xr(wsh_xr, msg='wse_to_wsh_xr check', assert_partial=assert_partial)
    return wsh_xr


def coarsen_dataarray(da, target_da, log=None, boundary='exact'):
    """coarsen with excplicit nodata handling"""
    plog = lambda msg: None if log is None else log.debug(msg)
    
    plog(f'coarsen_dataarray from {da.shape} to {target_da.shape}')
    assert_xr_geoTiff(da, msg='raw')
    assert_xr_geoTiff(target_da, msg='target')
    
    
    # Get the nodata value
    nodata_value = da.rio.nodata

    # Create a mask where the data is not equal to the nodata value
 
    # Apply the mask to the DataArray to create a masked array
    masked_da = da.where(da != nodata_value)

    # Calculate the coarsening factors for each dimension
    factors = [int(old / new) for old, new in zip(masked_da.shape, target_da.shape)]

    # Create a dictionary for the coarsen method
    coarsen_dict = {dim: factor for dim, factor in zip(masked_da.dims, factors)}

    # Perform the coarsening operation on the masked array
    coarsened_da = masked_da.coarsen(coarsen_dict).mean()

    # Convert the masked array back to an unmasked array by replacing np.nan with the nodata value
    coarsened_da = coarsened_da.fillna(nodata_value)
    
    #snap it back onto the reference
    da1 = coarsened_da.rio.reproject_match(target_da)
    #da1 = reproject_to_match(coarsened_da, target_da)
 
    assert_equal_raster_metadata(da1, target_da, msg='coarsen result')

    return da1







