'''
Created on May 25, 2024

@author: cef
'''
import os
import xarray as xr
import numpy as np
import numpy.ma as ma
from fdsc.assertions import assert_xr_geoTiff

def xr_to_GeoTiff(da, raster_fp, log=None, compress='LZW'): 

    #directory setup
    if not os.path.exists(os.path.dirname(raster_fp)):
        os.makedirs(os.path.dirname(raster_fp))
    
    assert raster_fp.endswith('.tif')
    
    assert isinstance(da, xr.DataArray), type(da)
    #===========================================================================
    # prep
    #===========================================================================
    da = da.fillna(-9999).rio.write_nodata(-9999)
    
    assert_xr_geoTiff(da)
    
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
    da1 =  da.fillna(nodata).rio.reproject_match(target_da, nodata=nodata, **kwargs)    
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