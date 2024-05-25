'''
Created on May 25, 2024

@author: cef

testing the main script
'''

import os, pathlib, pytest, logging, sys



@pytest.mark.parametrize('phase', ['00_raw'])
@pytest.mark.parametrize('caseName',[
    #'case_01', #no DEM
    #'case_ahr',
    #'case_jordan',
    'case_toy1',
    ])
@pytest.mark.parametrize('method, params',[
                         ('CostGrow', {}),
                         ])
def test_downscale_wse_raster(dem_fine_fp, wse_coarse_fp, 
                              method, params,
                              tmpdir, logger):
    from fdsc.main import downscale_wse_raster as func
    
    func(dem_fine_fp, wse_coarse_fp, 
         method=method, params=params,
         out_dir=tmpdir,logger=logger,
         ) 
    