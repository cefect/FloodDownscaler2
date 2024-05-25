'''
Created on May 25, 2024

@author: cef
'''

import xarray as xr
import numpy as np
import numpy.ma as ma


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
    )

    # Ensure rio attributes are copied over
 
    return new_dataarray.rio.write_crs(target_dataarray.rio.crs).rio.write_nodata(nodata)