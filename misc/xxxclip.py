'''
Created on Oct. 24, 2023

@author: cefect
'''

import os, argparse
from fdsc.control import Dsc_Session

def clip(
    dem_fp,
    wse_fp,
    aoi_fp,  
    out_dir=None,
    ):
    """cleanly clip a WSE and DEM grid using an AOI"""
    
    if out_dir is None:
        from definitions import wrk_dir
        out_dir= wrk_dir
        
    with Dsc_Session(run_name='fdsc2', relative=True, out_dir=out_dir,
                     aoi_fp=aoi_fp) as ses:
    
        ses.p0_clip_rasters(dem_fp, wse_fp)
        
        
if __name__ == "__main__":
    
    aoi_rp = r'L:/10_IO/2207_dscale/ins/ahr/aoi13/aoi09t_zoom0308_4647.geojson'
    
    clip(
        r'L:/10_IO/2207_dscale/ins/ahr/aoi13/r04/rim2d/dem005_r04_aoi13_0415.asc',
        r'L:/10_IO/2207_dscale/ins/ahr/aoi13/fdsc/r32_b10_i65_0511/wd_max_WSE.tif',
        aoi_rp
        )