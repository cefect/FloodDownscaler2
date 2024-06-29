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
              use_relative=True,
              ):
    """build a gdal vrt from a collection of files
    
    params
    --------------
    use_relative: bool, default True
        for portability, its usually best to use relative paths
        
    """
    #===========================================================================
    # defaults
    #===========================================================================
    if log is None: log=logging.getLogger('r')
    log = log.getChild('gdal.BuildVRT')
    
    #===========================================================================
    # gdal config
    #===========================================================================
    def gdal_log_callback(pct, message, unknown):
        if message is not None:
            log.info(f"{pct*100:.2f}% - {message}")
            
    
    def gdal_error_handler(err_class, err_no, err_msg):
        if err_class == gdal.CE_Warning:
            log.warning(f"GDAL Warning: {err_msg}")
        elif err_class == gdal.CE_Failure:
            log.error(f"GDAL Error: {err_msg}")
        else:
            log.info(f"GDAL Info: {err_msg}")
            
    
    gdal.PushErrorHandler(gdal_error_handler)  # Register the custom error handler
            
            
    if vrt_options is None:
        vrt_options = gdal.BuildVRTOptions(
                  separate=False, #whether each source file goes into a separate stacked band in the VRT band. 
                  strict=False, #set to True if warnings should be failures
                  #overwrite=True,
                  callback=gdal_log_callback,
                  callback_data=log,
                  )
        
    og_dir = os.getcwd()
    #===========================================================================
    # precheck
    #===========================================================================
    for i, fp in enumerate(fp_l):
        assert os.path.exists(fp), f'bad filepath on {i}/{len(fp_l)}\n    {fp}'
        
    #ensure teh paths are resolved
 
    if use_relative:
        vrt_dir = os.path.dirname(ofp)
        fp_l = [os.path.relpath(fp, vrt_dir) for fp in fp_l]
        
        os.chdir(vrt_dir)
        
        #[apath(f) for f in fp_l]
    
    #===========================================================================
    # # create a vrt raster using gdal from the filenames in 'fp_l'
    #===========================================================================
    log.debug(f' gdal.BuildVRT on {len(fp_l)} files to \n    {ofp}')
    
    vrt = gdal.BuildVRT(ofp,fp_l, options=vrt_options)
    if vrt is None:
        raise IOError(f'gdal.BuildVRT failed to generate a result')
    vrt.FlushCache()
    
    log.debug(f'finished on \n    {ofp}')
    os.chdir(og_dir)
    return ofp













