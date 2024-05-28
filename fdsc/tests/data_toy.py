'''
Created on Dec. 5, 2022

@author: cefect

toy test data
'''
import pytest, os
import numpy as np
import pandas as pd
import numpy.ma as ma
import shapely.geometry as sgeo
from pyproj.crs import CRS
from io import StringIO

import rasterio
from rasterio.transform import from_bounds

import xarray as xr
 
nan, array = np.nan, np.array

epsg = 3857
crs_default = CRS.from_user_input(epsg)

"""setup to construct when called
seems cleaner then building each time we init (some tests dont need these)
more seamless with using real data in tests"""

#===============================================================================
# helpers
#===============================================================================
#===============================================================================
# from fdsc.hp.rasters import (
#     get_mar, get_ar_from_str, get_rlay_fp,crs_default, 
#     )
# #from hp.hyd import get_wsh_ar
# from fdsc.hp.np import apply_block_reduce2, get_support_ratio
#===============================================================================


def get_ar_from_str(ar_str, dtype=float):
    return pd.read_csv(StringIO(ar_str), sep=',', header=None).astype(dtype).values



#===============================================================================
# raw data
#===============================================================================
test_data_lib = {
    'case_toy1':{
        'dem_fine_ar':
                get_ar_from_str("""
                        1,1,1,9,9,9,9,9,9
                        1,1,1,9,9,9,9,9,9
                        1,1,1,2,2,9,9,9,9
                        2,2,2,9,2,9,9,9,9
                        6,2,2,9,2,9,9,9,9
                        2,2,2,9,9,9,9,9,9
                        4,4,4,2,2,9,9,9,9
                        4,4,4,9,9,9,9,9,9
                        4,4,4,9,1,1,9,9,-9999
                        """),
        
        'wse_coarse_ar':array([
                                    [3.0, -9999, -9999],
                                    [4.0, -9999, -9999],
                                    [5.0, -9999, -9999],
                                    #[nan, nan, nan, nan]
                                    ]
                                ),
        '02_wp':{
            'wse_fine_xr':xr.DataArray(
                                np.array([
                                            [3.0, 3.0, 3.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                                            [3.0, 3.0, 3.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                                            [3.33333333, 3.33333333, 3.33333333, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                                            [3.66666667, 3.66666667, 3.66666667, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                                            [np.nan, 4.0, 4.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                                            [4.33333333, 4.33333333, 4.33333333, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                                            [4.66666667, 4.66666667, 4.66666667, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                                            [5.0, 5.0, 5.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                                            [5.0, 5.0, 5.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
                                        ]),
                                dims=["y", "x"],  # Note: y is the first dimension to match your example
                                coords={
                                    "x": np.arange(0.5, 9.5, 1),
                                    "y": np.arange(8.5, -0.5, -1),
                                    "band": 1,  # Single band
                                    "spatial_ref": 0,
                                },
                                attrs={
                                    "_FillValue": -9999.0
                                }
                            )
            }
        
        #=======================================================================
        # 'wsh_coarse_xr':ar_to_DataArray(  
        #                         array([
        #                             [2.0, 0.0, 0.0],
        #                             [2.0, 0.0, 0.0],
        #                             [1.0, 0.0, 0.0],
        #                             #[0.0, 0.0, 0.0, 0.0]
        #                             ])
        #                         )
        #=======================================================================
                }
    }

#get bounds from test DEM
s = test_data_lib['case_toy1']['dem_fine_ar'].shape
bbox_default = sgeo.box(0, 0, s[1], s[0]) #(0.0, 0.0, 6.0, 12.0)

def ar_to_geoTiff(array, file_path, 
                  crs=crs_default, 
                  bbox=bbox_default, 
                  nodata=-9999):
    """
    Writes a GeoTIFF file from a NumPy array.

 
    """
    assert file_path.endswith('tif')
    
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))
    
    # Validate input array shape (assuming single-band raster)
    if len(array.shape) != 2:
        raise ValueError("Input array should be 2D for a single-band raster.")
 
    with rasterio.open(
        file_path,
        'w',
        driver='GTiff',
        height=array.shape[0],
        width=array.shape[1],
        count=1,  # Single band
        dtype=array.dtype,
        crs=crs,
        transform = from_bounds(*bbox.bounds, array.shape[1], array.shape[0]),
        nodata=nodata,
    ) as dst:
        dst.write(array, 1)  # Write the array to the first (and only) ban
        
    return file_path


