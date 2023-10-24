'''
Created on Oct. 24, 2023

@author: cefect

CLI caller for running downscalers
'''
import os, argparse
from fdsc.control import Dsc_Session

def downscale(
    dem_fp,
    wse_fp,            
    
    method='CostGrow', 
    write_meta=True,
    
    out_dir=None,
    logger=None,
    
    **kwargs):
        """dowscale a coarse WSE grid
        
        Pars
        ----------
        wse_fp: str
            filepath to WSE raster layer at low-resolution (to be downscaled)
            
        dem_fp: str
            filepath to DEM raster layer at high-resolution (used to infer downscaled WSE)
            
        method: str
            downsccaling method to apply
            
        kwargs: dict
            key word arguments to pass to downscaling method
            
        write_meta: bool
            flag to write metadata"""
            
            
        if out_dir is None:
            from definitions import wrk_dir
            out_dir= wrk_dir
            
        with Dsc_Session(run_name='fdsc2', relative=True, out_dir=out_dir, logger=logger) as ses:
 
            return ses.run_dsc(dem_fp, wse_fp, method=method, write_meta=write_meta,
                        rkwargs=kwargs,
                        )
            
 

 

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
