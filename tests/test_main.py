'''
Created on May 25, 2024

@author: cef

testing the main script
'''

import os, pathlib, pytest, logging, sys

from .conftest import write_to_test_data
from definitions import test_data_dir

#print(f"Script path: {os.path.abspath(__file__)}\n Script directory: {os.path.dirname(os.path.abspath(__file__))}\n Current working directory: {os.getcwd()}")

import numpy as np
np.set_printoptions(linewidth=300)


@pytest.mark.parametrize('phase', ['00_raw'])
@pytest.mark.parametrize('caseName',[
    'case_toy1',
    
    'case_ahr', #(16, 18) to (128, 144)
    'case_jordan', #(197, 213) to (591, 639) EPSG4326    
    #'case_f3n2e100', #EPSG:4326. 9000x9000, 3:1. slow
    pytest.param('case_ruth', marks=pytest.mark.xfail)
 
    ])
@pytest.mark.parametrize('method, params',[
                         ('CostGrow', dict(distance_fill='neutral')),
                         #('CostGrow', dict(distance_fill='terrain_penalty')),
                         ])
def test_downscale_wse_raster(dem_fine_fp, wse_coarse_fp, 
                              method, params,
                              tmpdir, logger, caseName):
    
    print(f'caseName: {caseName}')
    from fdsc.main import downscale_wse_raster as func
    
    """no.. not setup for this
    if write_to_test_data:
        out_dir = os.path.join(test_data_dir, caseName)
    else:
        out_dir = tmpdir"""
    
    
    func(dem_fine_fp, wse_coarse_fp, 
         method=method, 
         out_dir=os.path.join(tmpdir, caseName),
         logger=logger,
         pluvial=False,
         **params) 
    
    
@pytest.mark.dev
@pytest.mark.parametrize('phase', ['00_raw'])
@pytest.mark.parametrize('caseName',[ 
    'case_ruth',
    ])
@pytest.mark.parametrize('method, params',[
                         ('CostGrow', dict(distance_fill='neutral', wsh_coarse_thresh=0.2)),
                         #('CostGrow', dict(distance_fill='terrain_penalty')),
                         ])
@pytest.mark.parametrize('use_wsh', [
    #True, 
    False])
def test_downscale_pluvial_wse_raster(dem_fine_fp, wse_coarse_fp, 
                                      wsh_coarse_fp,
                              method, params,use_wsh,
                              tmpdir, logger, caseName):
    
    print(f'caseName: {caseName}')
    from fdsc.main import downscale_wse_raster as func
    
    if not use_wsh:
        wsh_coarse_fp = None
    
    func(dem_fine_fp, wse_coarse_fp, 
         method=method,  
         out_dir=os.path.join(tmpdir, caseName),
         logger=logger,
         pluvial=True, wsh_coarse_fp=wsh_coarse_fp,
         **params) 