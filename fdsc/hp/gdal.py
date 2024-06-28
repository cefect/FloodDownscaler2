'''
Created on Jun. 2, 2024

@author: cef
'''

import logging, os
from osgeo import gdal
from .dirz import make_dir

from pathlib import PureWindowsPath, Path
apath = lambda x:str(Path(x).resolve())


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
    
    
    
    
def build_vrt(ofp, fp_l, log=None,
              vrt_options = None,
              ):
    #===========================================================================
    # defaults
    #===========================================================================
    if log is None: log=logging.getLogger('r')
    log = log.getChild('gdal.BuildVRT')
    
    def gdal_log_callback(pct, message, unknown):
        if message is not None:
            log.info(f"{pct*100:.2f}% - {message}")
            
            
    if vrt_options is None:
        vrt_options = gdal.BuildVRTOptions(
                  separate=False, #whether each source file goes into a separate stacked band in the VRT band. 
                  strict=False, #set to True if warnings should be failures
                  #overwrite=True,
                  )
        
    #===========================================================================
    # precheck
    #===========================================================================
    for i, fp in enumerate(fp_l):
        assert os.path.exists(fp), f'bad filepath on {i}/{len(fp_l)}\n    {fp}'
        
    #ensure teh paths are resolved
    
    #===========================================================================
    # # create a vrt raster using gdal from the filenames in 'fp_l'
    #===========================================================================
    log.debug(f' on {len(fp_l)} files')
    
    vrt = gdal.BuildVRT(ofp,[apath(f) for f in fp_l], options=vrt_options)
    vrt.FlushCache()
    
    log.debug(f'finished on \n    {ofp}')
    return ofp