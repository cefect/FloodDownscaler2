'''
Created on Oct. 24, 2023

@author: cefect

CLI caller for running downscalers


2024-05-25:
    re-write to xarray
    xarray has limited support for masks/nulls
'''

#===============================================================================
# imports----------
#===============================================================================
import os, argparse, logging
import rioxarray

from parameters import today_str
from .hp.logr import get_new_file_logger, get_log_stream
from .hp.rio import geographic_to_projected
from .assertions import *


#===============================================================================
# HELPERS--------
#===============================================================================
def _geoTiff_to_xr(fp, nodata=-9999): 
    return rioxarray.open_rasterio(fp,masked=True).squeeze().rio.write_nodata(nodata)

 

#===============================================================================
# RUNNERS-----------
#===============================================================================
def downscale_wse_raster(
    dem_fine_fp,
    wse_coarse_fp,            
    
    method='CostGrow', 
    params=None,
    write_meta=True,
    
    out_dir=None,
    logger=None,
 
    
    **kwargs):
    """dowscale a coarse WSE grid using a fine DEM from raster files
    
    Pars
    ----------
    wse_coarse_fp: str
        filepath to WSE raster layer at coarse-resolution (to be downscaled)
        
    dem_fine_fp: str
        filepath to DEM raster layer at fine-resolution (used to infer downscaled WSE)
        
    method: str
        downsccaling method to apply
            CostGrow                
        
    params: dict
        key word arguments to pass to downscaling method
        
    write_meta: bool
        flag to write metadata
        
        
    """
        
    #=======================================================================
    # defaults
    #=======================================================================
    if out_dir is None:
        from definitions import wrk_dir
        out_dir= wrk_dir
        
    if logger is None:
        logger = get_new_file_logger(
                    fp = os.path.join(out_dir, f'downscale_wse_raster_{today_str}.log'),
                    level = logging.DEBUG,
                    logger=get_log_stream(level = logging.DEBUG)
                    )
        
    
    #===========================================================================
    # pre-process
    #===========================================================================
    """this is a bad idea... no good way to revert after compute
    dem_fine_fp = geographic_to_projected(dem_fine_fp)
    """
    
    #===========================================================================
    # load to Xarray
    #===========================================================================
    dem_fine_xr = _geoTiff_to_xr(dem_fine_fp)
    assert_xr_geoTiff(dem_fine_xr)
        
    wse_coarse_xr = _geoTiff_to_xr(wse_coarse_fp)
    assert_xr_geoTiff(wse_coarse_xr)
    
    
    
    logger.info(f'enahcing resolution from {wse_coarse_xr.shape} to {dem_fine_xr.shape}')
    #===========================================================================
    # execute
    #===========================================================================
    if method=='CostGrow':
        from fdsc.alg.costGrow import downscale_costGrow_xr as func
        
    else:
        raise KeyError(method)
    
    
    wse_fine_xr, meta_d = func(dem_fine_xr, wse_coarse_xr,logger=logger,
                               write_meta=write_meta, out_dir=out_dir, **params)
    
    
 
            
 

 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Downscale a coarse WSE grid.')
    parser.add_argument('dem_fp', type=str, help='Filepath to DEM raster layer at fine-resolution.')
    parser.add_argument('wse_fp', type=str, help='Filepath to WSE raster layer at coarse-resolution.')
    parser.add_argument('--method', type=str, default='CostGrow', help='Downscaling method to apply.')
    parser.add_argument('--write_meta', type=bool, default=True, help='Flag to write metadata.')
    parser.add_argument('--out_dir', type=str, default=None, help='Output directory.')
    args, unknown_args = parser.parse_known_args()
    
    # parse unknowns
    kwargs = dict()
    k = None
    for e in unknown_args:
        # keys
        if e.startswith('-'):
            k = e.replace('-', '')
            kwargs[k] = None
            
        # values
        else:
            assert not k is None
            kwargs[k] = e
            k = None

 
    downscale(args.dem_fp, args.wse_fp, method=args.method, write_meta=args.write_meta, out_dir=args.out_dir, **kwargs)
