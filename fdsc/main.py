'''
Created on Oct. 24, 2023

@author: cefect

CLI caller for running downscalers
'''
import os
from fdsc.control import Dsc_Session

def downscale(
    dem1_fp,
    wse2_fp,            
    
    method='CostGrow', 
    write_meta=True,
    
    out_dir=None,
    
    **kwargs):
        """dowscale a coarse WSE grid
        
        Pars
        ----------
        wse2_fp: str
            filepath to WSE raster layer at low-resolution (to be downscaled)
            
        dem1_fp: str
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
            
        with Dsc_Session(run_name='fdsc2', relative=True, out_dir=out_dir) as ses:
 
            ses.run_dsc(dem1_fp, wse2_fp, method=method, write_meta=write_meta,
                        rkwargs=kwargs,
                        )
            
            
 