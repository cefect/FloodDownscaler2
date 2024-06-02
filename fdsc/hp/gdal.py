'''
Created on Jun. 2, 2024

@author: cef
'''

 
from osgeo import gdal


def calculate_slope(input_dem, output_slope, scale=1, slope_format='percent'):
    """
    Calculates slope from a DEM using GDAL.

    Args:
        input_dem (str): Path to the input DEM raster file.
        output_slope (str): Path to the output slope raster file.
        scale (float, optional): Ratio of vertical units to horizontal units. Defaults to 1.
        slope_format (str, optional): Slope format ('percent' or 'degrees'). Defaults to 'percent'.
    """
    
    # Open the DEM
    dem_ds = gdal.Open(input_dem)

    # Run the GDAL DEM slope algorithm
    gdal.DEMProcessing(output_slope, dem_ds, 'slope', format=slope_format, scale=scale)

    # Close the datasets
    dem_ds = None  # Dereference the dataset object 