'''
Created on May 25, 2024

@author: cef
'''

import datetime, os, tempfile, warnings
import pandas as pd
import numpy as np
import numpy.ma as ma
import rasterio as rio
import rasterio.features

import xarray as xr




def assert_mask_fp(rlay_fp,
               msg='',
                **kwargs):
    """check the passed rlay is a mask-like raster"""
    #assertion setup
    if not __debug__: # true if Python was not started with an -O option
        return
    __tracebackhide__ = True    

    assert isinstance(rlay_fp, str)
    assert os.path.exists(rlay_fp)
    
    #need to use the custom loader. this calls assert_mask_ar
    try:
        load_mask_array(rlay_fp, **kwargs)
    except Exception as e:
        raise TypeError(f'{e}\n    not a mask: '+msg)
    #rlay_ar_apply(rlay, assert_mask_ar, masked=False, **kwargs)
    

    
def assert_mask_ar_raw(ar,  maskType='binary', msg=''):
    """check raw raster array conforms to our mask speecifications
    
    usually we deal with processed masks... see assert_mask_ar
    
    see load_mask_array
    
    Parameters
    --------------
    maskType: str
        see load_mask_array
    """
    if not __debug__: # true if Python was not started with an -O option
        return 
    __tracebackhide__ = True
    
    #===========================================================================
    # get test function based on mask typee
    #===========================================================================
    
    if maskType=='binary':
        assert not isinstance(ar, ma.MaskedArray)
        vals = set(np.unique(ar.ravel()))
        evals = {0.0, 1.0}        
        
    elif maskType=='native':
        assert isinstance(ar, ma.MaskedArray)
        """sets don't like the nulls apparently"""
        vals = set(np.unique(np.where(ar.mask, 1.0, ar.data).ravel()))
        
        #raise NotImplementedError('not sure about this one....')
        evals = {1.0}  
    else:
        raise KeyError(maskType)
    
 
    if not vals.symmetric_difference(evals)==set():
        raise AssertionError(f'got unexpected values for maskType {maskType}\n    {vals}')
    

def assert_masked_ar(ar, msg=''):
    """check the array satisfies expectations for a masked array
        not to be comfused with a MASK array
     
    NOTE: to call this on a raster filepath, wrap with rlay_ar_apply:
        rlay_ar_apply(wse1_dp_fp, assert_wse_ar, msg='result WSe')
    """
    if not __debug__: # true if Python was not started with an -O option
        return
     
    if not isinstance(ar, ma.MaskedArray):
        raise AssertionError(msg+'\n     bad type ' + str(type(ar)))
    if not 'float' in ar.dtype.name:
        raise AssertionError(msg+'\n     bad dtype ' + ar.dtype.name)
     
    #check there are no nulls on the data
    if np.any(np.isnan(ar.filled())):
        raise AssertionError(msg+f'\n    got {np.isnan(ar.data).sum()}/{ar.size} nulls outside of mask')
    
    if np.all(ar.mask):
        raise AssertionError(msg+f'\n    passed array is fully masked')
         
  
    
def assert_mask_ar(ar, msg=''):
    """check if mask array
        not to be confused with MASKED array
   
    TODO: accept masks (we should be able to keep the mask information)
    
    see load_mask_array
    """
    if not __debug__: # true if Python was not started with an -O option
        return 
    __tracebackhide__ = True
    
    assert not isinstance(ar, ma.MaskedArray), msg
    assert ar.dtype==np.dtype('bool'), msg


def assert_partial_wet(ar, msg=''):
    """assert a boolean array has some trues and some falses (but not all)"""
    if not __debug__: # true if Python was not started with an -O option
        return
    #__tracebackhide__ = True 
    
    #assert isinstance(ar, ma.MaskedArray)
    assert 'bool' in ar.dtype.name
    
    if np.all(ar):
        raise AssertionError(msg+': all masked/wet cells')
    
    if np.all(np.invert(ar)):
        raise AssertionError(msg+': no mask/dry cells')

def assert_inun_ar(ar, msg=''):
    """inundation array. wet=True"""
    if not __debug__:  
        return 
    #__tracebackhide__ = True
    
    assert_mask_ar(ar, msg=msg+' inun')
    if not ar.any():
        raise AssertionError(f'expect some Trues\n'+msg)
    
      


def assert_dem_ar(ar, msg=''):
    """check the array satisfies expectations for a DEM array"""
    if not __debug__: # true if Python was not started with an -O option
        return
    __tracebackhide__ = True 
    
    assert len(ar.shape)==2
    assert_masked_ar(ar, msg=msg)
    
    """this is OK.. dem can have some nulls
    if not np.all(np.invert(ar.mask)):
        raise AssertionError(msg+': some masked values')"""
    
def assert_wse_ar(ar, msg='', assert_partial=True):
    """check the array satisfies expectations for a WSE array"""
    if not __debug__: # true if Python was not started with an -O option
        return    
    __tracebackhide__ = True   
    
    try:
        assert_masked_ar(ar)    
        if assert_partial:
            assert_partial_wet(ar.mask)
    except Exception as e:
        raise TypeError(msg+f'\npassed array does not conform to WSE expectations\n{e}')
        
    
def assert_wsh_ar(ar, msg='', assert_partial=True):
    """check the array satisfies expectations for a WD array"""
    if not __debug__: # true if Python was not started with an -O option
        return
    
    assert_masked_ar(ar, msg=msg)
    
    """this is OK... WSH mask can align with teh DEM mask
    if not np.all(np.invert(ar.mask)):
        raise AssertionError(msg+'\n got some masked values on a WSH grid')
    """
    if assert_partial:
        if not np.isclose(np.min(ar), 0.0, atol=1e-4):
            raise AssertionError(msg+': expected zero minimum, got %.6f'%np.min(ar)) 
    
    if not np.max(ar)>0.0:
        raise AssertionError(msg+': got max=0 on WSH (ie all dry)')
    
    if np.max(ar)>1e5:
        raise AssertionError(msg+'/nexcessive depths: {np.max(ar)}') 
    
    
    
 
def assert_equal_extents_xr(left_xr, right_xr, msg='', rtol=1e-5, atol=1e-8):
    """
    Asserts that two xarray DataArrays have approximately equal extents.

    Args:
        left_xr (xr.DataArray): The first xarray DataArray to compare.
        right_xr (xr.DataArray): The second xarray DataArray to compare.
        msg (str, optional): An additional message to include in the AssertionError.

    Raises:
        AssertionError: If the extents are not approximately equal.
    """
    if not __debug__:  # Skip assertion in optimized mode (-O)
        return
    
    assert_xr_geoTiff(left_xr)
    assert_xr_geoTiff(right_xr)
    
 

    # Get bounding boxes as tuples (minx, miny, maxx, maxy)
    left_bounds = left_xr.rio.bounds()
    right_bounds = right_xr.rio.bounds()

    # Check for approximate equality of bounding box values
    if not np.allclose(left_bounds, right_bounds, rtol=rtol, atol=atol):
        raise AssertionError(f"Bounding box extents are not equal. {msg}")

def assert_xr_geoTiff(da, nodata_value=-9999, x_dim="x", y_dim="y",
                      msg=''):
    """
    Asserts that a rioda DataArray conforms to typical GeoTIFF conventions.

    Args:
        da (xr.DataArray): The rioda DataArray to check.
        nodata_value (int, optional): The expected nodata value (default: -9999).
        x_dim (str, optional): The name of the x dimension (default: "x").
        y_dim (str, optional): The name of the y dimension (default: "y").

    Raises:
        AssertionError: If the DataArray does not conform to the conventions.
    """
    if not __debug__:  # Skip assertion in optimized mode (-O)
        return

    if not isinstance(da, xr.DataArray):
        raise AssertionError("Input must be an da DataArray.")
    
    
    #NO... need to preserve the data structure...
    #reprojection introduces more problems than it solves
    #===========================================================================
    # if not da.rio.crs.is_geographic:
    #     """most things would work fine in geographic, 
    #     but this introduces some complexity when computing horizontal vs vertical distances I'd rather not htink about
    #     """
    #     raise AssertionError(f'expects a projected crs. reproject your layers')
    #===========================================================================
    
    #need to allow this for geographic data
    try:
        assert_square_pixels(da, msg=msg)
    except Exception as e:
        warnings.warn(f'non-square pixels\n{e}')

    # Check nodata value
    if da.rio.nodata != nodata_value:
        raise AssertionError(
            f"Nodata value does not match expected value: {da.rio.nodata} != {nodata_value}\n"+msg
        )
        
    if da.rio.crs is None:
        raise AssertionError(f'CRS must be assigned\n    {msg}')

    # Check for presence of x and y dimensions
    if x_dim not in da.dims or y_dim not in da.dims:
        raise AssertionError(f"Missing required dimensions: {x_dim} or {y_dim}")
    
    # Check that x and y dims are 1D
    if len(da[x_dim].shape) != 1 or len(da[y_dim].shape) != 1:
        raise AssertionError(f"Dimensions '{x_dim}' and '{y_dim}' must be 1D")
    
    
def assert_square_pixels(xarray, rtol=1e-5, msg=''):
    """
    Asserts that a rioxarray DataArray has approximately square pixels.

    Args:
        xarray (xr.DataArray): The rioxarray DataArray to check.
        rtol (float, optional): Relative tolerance for pixel dimension comparison (default: 1e-5).
        msg (str, optional): An additional message to include in the AssertionError.

    Raises:
        AssertionError: If the DataArray does not have approximately square pixels.
    """
    if not __debug__:  # Skip assertion in optimized mode (-O)
        return

    if not isinstance(xarray, xr.DataArray):
        raise AssertionError("Input must be an xarray DataArray.")
    
    # Check if resolution exists for both x and y dimensions
    if xarray.rio.resolution() is None:
        raise AssertionError("Resolution information not found in the xarray DataArray.")
    
    # Extract x and y resolution
    x_res, y_res = xarray.rio.resolution()
    
    """normal for geographic data
    if not x_res>0 or y_res>0:
        raise AssertionError(f'expect positive resolution, instead got {x_res} x {y_res}\n    {msg}')
        
    """

    # Check for approximate equality of x and y resolutions
    if not np.isclose(abs(x_res), abs(y_res), rtol=rtol):
        raise AssertionError(f"Pixel dimensions {x_res} x {y_res} are not approximately square. \n{msg}")

 
def assert_dem_xr(dem_xr, msg=''):
    assert_xr_geoTiff(dem_xr, msg=msg) 
    return assert_dem_ar(dem_xr.to_masked_array(), msg=msg)

def assert_wse_xr(wse_xr, msg='', **kwargs):
    assert_xr_geoTiff(wse_xr, msg=msg)
    return assert_wse_ar(wse_xr.to_masked_array(), msg=msg, **kwargs)

def assert_wsh_xr(da, msg='', **kwargs):
    assert_xr_geoTiff(da, msg=msg)
    return assert_wsh_ar(da.to_masked_array(), msg=msg, **kwargs)


def assert_wse_vs_dem_mar(wse_mar, dem_mar, msg='', **kwargs):
    assert_wse_ar(wse_mar, msg=msg, **kwargs)
    assert_dem_ar(dem_mar, msg=msg)
    
    #===========================================================================
    # check WSE logic
    #===========================================================================
    delta_mar = ma.MaskedArray(
        wse_mar.data - dem_mar.data, mask=dem_mar.mask)
    bool_ar = delta_mar[~delta_mar.mask].ravel() < 0.0
    if bool_ar.any():
        raise AssertionError(f'{bool_ar.sum()}/{bool_ar.size} WSE pixels were at or below the DEM\n' + msg)

def assert_wse_vs_dem_xr(wse_xr, dem_xr, msg='', **kwargs):
    """check consistency between the WSE and the DEM"""
    
    if not __debug__: # true if Python was not started with an -O option
        return    
    __tracebackhide__ = True  
    
    #basic data checks
    assert_equal_raster_metadata(wse_xr, dem_xr, msg=msg)
    
    wse_mar= wse_xr.to_masked_array()
    dem_mar = dem_xr.to_masked_array()


    assert_wse_vs_dem_mar(wse_mar, dem_mar, msg=msg, **kwargs)
    
 


def assert_integer_like_and_nearly_identical(arr, rtol=1e-3, atol=1e-3):
    """
    Asserts that all values in a NumPy array are:

    1. Close to integers (within a tolerance)
    2. Nearly identical to each other (within a tolerance)

    Args:
        arr (np.ndarray): The NumPy array to check.
        rtol (float, optional): Relative tolerance for integer comparison (default: 1e-5).
        atol (float, optional): Absolute tolerance for integer and equality comparisons (default: 1e-8).

    Raises:
        AssertionError: If any value is not close to an integer or if values are not nearly identical.
    """

    if not __debug__:  # Skip assertion in optimized mode (-O)
        return

    # Check if all values are close to integers
    if not np.allclose(arr, arr.astype(int), rtol=rtol, atol=atol):
        raise AssertionError(
            f"Not all values in the array are close to integers within tolerance (rtol={rtol}, atol={atol})."
        )

    # Check if all values are nearly identical
    if not np.allclose(arr, arr.mean(), rtol=rtol, atol=atol):
        raise AssertionError(
            f"Values in the array are not nearly identical within tolerance (rtol={rtol}, atol={atol})."
        )


 

def assert_equal_raster_metadata(left_xr, right_xr, coord_tolerance=1e-6, msg=""):
    """
    Asserts that two rioxarray DataArrays have identical spatial and rasterio metadata,
    including near-identical coordinates within a tolerance.

    Args:
        left_xr (xr.DataArray): The first rioxarray DataArray to compare.
        right_xr (xr.DataArray): The second rioxarray DataArray to compare.
        coord_tolerance (float, optional): The tolerance for coordinate differences (default 1e-6).
        msg (str, optional): An additional message to include in the AssertionError.

    Raises:
        AssertionError: If any of the spatial or rasterio attributes differ,
                        including coordinates beyond the tolerance.
    """
    if not __debug__:  # Skip assertion in optimized mode (-O)
        return
    
    if not isinstance(left_xr, xr.DataArray) or not isinstance(right_xr, xr.DataArray):
        raise AssertionError(
            "Both inputs must be xarray DataArrays. Types found were: "
            f"{type(left_xr)} and {type(right_xr)}"
        )

    # Check dimensions
    if left_xr.dims != right_xr.dims:
        raise AssertionError(f"Dimension names do not match: {left_xr.rio.dims} vs {right_xr.rio.dims}." + msg)

    # Check coordinate values within tolerance
    if not np.allclose(left_xr.x, right_xr.x, atol=coord_tolerance):
        raise AssertionError(f"X coordinates do not match within tolerance ({coord_tolerance})." + msg)
    if not np.allclose(left_xr.y, right_xr.y, atol=coord_tolerance):
        raise AssertionError(f"Y coordinates do not match within tolerance ({coord_tolerance})." + msg)

    # Check coordinate reference system (CRS)
    if left_xr.rio.crs != right_xr.rio.crs:
        raise AssertionError(f"CRS does not match: {left_xr.rio.crs} vs {right_xr.rio.crs}." + msg)

    # Check transform
    if not np.allclose(left_xr.rio.transform(), right_xr.rio.transform()):
        raise AssertionError(f"Transform does not match: {left_xr.rio.transform()} vs {right_xr.rio.transform()}." + msg)

    # Check resolution
    if not np.allclose(left_xr.rio.resolution(), right_xr.rio.resolution()):
        raise AssertionError(f"Resolution does not match: {left_xr.rio.resolution()} vs {right_xr.rio.resolution()}." + msg)

    # Optionally check other attributes like nodata, count (number of bands)
    if left_xr.rio.nodata != right_xr.rio.nodata:
        raise AssertionError(f"Nodata value does not match: {left_xr.rio.nodata} vs {right_xr.rio.nodata}." + msg)

    if left_xr.rio.count != right_xr.rio.count:
        raise AssertionError(f"Number of bands does not match: {left_xr.rio.count} vs {right_xr.rio.count}." + msg)

    
    