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
from .coms import set_da_layerNames
from .hp.logr import get_new_file_logger, get_log_stream
from .hp.rio import geographic_to_projected
from .hp.xr import xr_to_GeoTiff
from .assertions import *


#===============================================================================
# HELPERS--------
#===============================================================================
def _geoTiff_to_xr(fp, nodata=-9999): 
    return rioxarray.open_rasterio(fp,masked=True).squeeze().compute().rio.write_nodata(nodata)

 

#===============================================================================
# RUNNERS-----------
#===============================================================================
def downscale_wse_raster(
    dem_fine_fp,
    wse_coarse_fp,            
    
    method='CostGrow', 
    pluvial=False, wsh_coarse_fp=None, dem_coarse_fp=None,
    #params=None,
    write_meta=True,
    
    out_dir=None, ofp=None,
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
        
    log=logger.getChild('main')
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
    wse_coarse_xr = _geoTiff_to_xr(wse_coarse_fp)    
    
    #add layerNames and do some checks
    set_da_layerNames({'dem_fine':dem_fine_xr,'wse_coarse': wse_coarse_xr})
    
 
    
    logger.info(f'enahcing resolution from {wse_coarse_xr.shape} to {dem_fine_xr.shape}')
    #===========================================================================
    # retrieve WSE projection method
    #===========================================================================
    
    
    
    if method=='CostGrow':
        from fdsc.alg.costGrow import downscale_costGrow_xr as wse_proj_func
        
    else:
        raise KeyError(method)
    
    #===========================================================================
    # set up pluvial wrapper
    #===========================================================================
    if not pluvial:
        func = wse_proj_func
    else:
        log.info('preparing pluvial downscaling')
        from fdsc.alg.pluvial import downscale_pluvial_xr as func
        
        #load the coarse WSH if present
        for layerName, fp in {
            'wsh_coarse':wsh_coarse_fp, 'dem_coarse':dem_coarse_fp,
            }.items():
        
            if not fp is None:
                log.debug(f'loading {layerName} from \n    {fp}')                
                da =  _geoTiff_to_xr(fp)                
                set_da_layerNames({layerName:da})
                kwargs[layerName+'_xr'] = da
 
        #add the wse projection to teh kwargs
        kwargs['wse_proj_func'] = wse_proj_func
 
    
    #===========================================================================
    # execute
    #===========================================================================
    wse_fine_xr, meta_d =  func(dem_fine_xr, wse_coarse_xr,logger=logger,
                               write_meta=write_meta, out_dir=out_dir, **kwargs)
    
    #===========================================================================
    # wrap
    #===========================================================================
    if ofp is None:
        ofp = os.path.join(out_dir, f'wse_downscaled_{method}.tif')
        xr_to_GeoTiff(wse_fine_xr, ofp, log=log)
        
    return ofp, meta_d
    
    
 
            
 

 

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

 
    downscale_wse_raster(args.dem_fp, args.wse_fp, method=args.method, write_meta=args.write_meta, out_dir=args.out_dir, **kwargs)
