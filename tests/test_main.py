'''
Created on May 25, 2024

@author: cef

testing the main script
'''

import os, pathlib, pytest, logging, sys

import numpy as np
np.set_printoptions(linewidth=300)

@pytest.mark.parametrize('phase', ['00_raw'])
@pytest.mark.parametrize('caseName',[
    'case_toy1',
    #'case_01', #no DEM
    #'case_ahr', #(16, 18) to (128, 144)
    #'case_jordan', #(197, 213) to (591, 639) EPSG4326    
    #'case_f3n2e100', #EPSG:4326. 9000x9000, 3:1
    ])
@pytest.mark.parametrize('method, params',[
                         ('CostGrow', dict(distance_fill='neutral')),
                         ('CostGrow', dict(distance_fill='terrain_penalty')),
                         ])
def test_downscale_wse_raster(dem_fine_fp, wse_coarse_fp, 
                              method, params,
                              tmpdir, logger, caseName):
    
    print(f'caseName: {caseName}')
    from fdsc.main import downscale_wse_raster as func
    
    func(dem_fine_fp, wse_coarse_fp, 
         method=method, params=params,
         out_dir=os.path.join(tmpdir, caseName),
         logger=logger,
         ) 
    