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
    #'case_jordan', #EPSG4326
    #'case_toy1',
    'case_f3n2e100', #EPSG:4326. 9000x9000, 3:1
    ])
@pytest.mark.parametrize('method, params',[
                         ('CostGrow', {}),
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
    