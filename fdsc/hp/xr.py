'''
Created on May 25, 2024

@author: cef
'''
import os
import xarray as xr
import numpy as np
import numpy.ma as ma
from rasterio.warp import transform

from ..assertions import assert_xr_geoTiff, assert_wsh_xr

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
    
    msg = f'wrote {da.shape} to \n    {raster_fp}'
    if not log is None:
        log.debug(msg)
    else:
        print(msg)
    
    return raster_fp

def resample_match_xr(da, target_da, **kwargs):
    """resampling with better treatment for masks"""
    
    
    #wse_coarse_xr.fillna(0.0).interp_like(dem_fine_xr, method='linear', assume_sorted=True, kwargs={'fill_value':'extrapolate'})
    nodata = da.rio.nodata
    da1 =  da.fillna(nodata).rio.reproject_match(
        target_da, nodata=nodata, **kwargs)    
    return da1.where(da1!=nodata, np.nan)



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

    # Get the nodata value from the target DataArray
    nodata = target_dataarray.rio.nodata

    # Replace masked values with nodata
    filled_array = ma.filled(masked_array, fill_value=np.nan)

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


def wse_to_wsh_xr(dem_xr, wse_xr, log=None):
    plog = lambda msg: None if log is None else log.debug(msg)
    
    log.debug(f'wse_to_wsh_xr on {wse_xr}')
    
    delta_ar = np.nan_to_num(wse_xr.data - dem_xr.data, 0.0)
    
    wsh_mar = ma.MaskedArray(np.where(delta_ar < 0.0, 0.0, delta_ar), 
        mask=dem_xr.to_masked_array().mask, #mask where DEM is masked
        )
 
    wsh_xr = dataarray_from_masked(wsh_mar, dem_xr)
    
    if 'layerName' in wse_xr.attrs:
        wsh_xr.attrs['layerName'] = wse_xr.attrs['layerName'].lower().replace('wse', 'wsh')
        
    assert_wsh_xr(wsh_xr, msg='wse_to_wsh_xr check')
    return wsh_xr










