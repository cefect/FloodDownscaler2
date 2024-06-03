'''
Created on May 25, 2024

@author: cef

testing the main script
'''

import os, pathlib, pytest, logging, sys, pprint

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
    #pytest.param('case_ruth', marks=pytest.mark.xfail) #pluvial
 
    ])
@pytest.mark.parametrize('method, params',[
                         ('CostGrow', dict(distance_fill='neutral', dp_coarse_pixel_max=2)),
                         ('CostGrow', dict(distance_fill='terrain_penalty', distance_fill_method='distance_transform_edt')),
                         ])
@pytest.mark.parametrize('decay_method_d',[
    {'linear':dict(decay_frac=0.005),'slope_linear':dict( decay_frac=0.005),'slope_power':dict(n=1.4, c=0.001)},
    {},    
    ])
def test_downscale_wse_raster(dem_fine_fp, wse_coarse_fp, 
                              method, params,decay_method_d,
                              tmpdir, logger, caseName):
    
    print(f'caseName: {caseName}')
    from ..fdsc.main import downscale_wse_raster as func
    
    """no.. not setup for this
    if write_to_test_data:
        out_dir = os.path.join(test_data_dir, caseName)
    else:
        out_dir = tmpdir"""
    
    
    func(dem_fine_fp, wse_coarse_fp, 
         method=method, 
         out_dir=os.path.join(tmpdir, caseName),
         logger=logger,
         pluvial=False, decay_method_d=decay_method_d,
         **params) 
    
    
@pytest.mark.dev
@pytest.mark.parametrize('phase', ['00_raw'])
@pytest.mark.parametrize('caseName',[ 
    'case_ruth',
    ])
@pytest.mark.parametrize('method, params',[ 
  ('CostGrow', dict(distance_fill='neutral',distance_fill_method='distance_transform_edt',dp_coarse_pixel_max=50)),
  ('CostGrow', dict(distance_fill='terrain_penalty',distance_fill_method='distance_transform_edt',dp_coarse_pixel_max=10)),
  ])
 
@pytest.mark.parametrize('pluvial_params',[
             #dict(reapply_small_groups=False, small_pixel_count=10, filter_depth=None, filter_depth_mode_buffer=0.2), #pretty aggressive
              dict(reapply_small_groups=False, small_pixel_count=5, filter_depth=None, filter_depth_mode_buffer=0.1),  
             #dict(reapply_small_groups=False, small_pixel_count=5, filter_depth=None)
                             ])
@pytest.mark.parametrize('decay_method_d', [
    #===========================================================================
    # {
    #     'linear':dict(decay_frac=0.01),
    #     #'slope_linear':dict(decay_frac=0.001), 
    #     'slope_power':dict(n=1.5, c=0.001) #looks pretty good!
    #     },
    #===========================================================================
        #=======================================================================
        # {
        # 'linear':dict(decay_frac=0.01),
        # #'slope_linear':dict(decay_frac=0.001), 
        # 'slope_power':dict(n=1.4, c=0.001) #also good.. a bit more spill
        # },
        # {
        # 'linear':dict(decay_frac=0.01),
        # #'slope_linear':dict(decay_frac=0.001), 
        # 'slope_power':dict(n=1.3, c=0.01) #to aggressive
        # },
        #=======================================================================
        #=======================================================================
        # {
        # 'linear':dict(decay_frac=0.01),
        # #'slope_linear':dict(decay_frac=0.001), 
        # 'slope_power':dict(n=1.3, c=0.001)  #too liberal
        # },
        #=======================================================================
        #=======================================================================
        #         {
        # 'linear':dict(decay_frac=0.005), #looks good
        # #'slope_linear':dict(decay_frac=0.001), 
        # 'slope_power':dict(n=1.4, c=0.001)   
        # },
        #=======================================================================
        #=======================================================================
        #     {
        # 'linear':dict(decay_frac=0.01), 'slope_power':dict(n=1.3, c=0.005) #too much spill   
        # },
        #=======================================================================
        #{'linear':dict(decay_frac=0.01), 'slope_power':dict(n=1.4, c=0.005)}, #too aggressive
         {'linear':dict(decay_frac=0.005), 'slope_power':dict(n=1.4, c=0.001)}, #looks nice
    ])
@pytest.mark.parametrize('use_wsh', [
    True, 
    #False,
    ])
@pytest.mark.parametrize('use_dem', [
    True, #nice for writing some coarse debug outouts 
    #False,
    ])
def test_downscale_pluvial_wse_raster(dem_fine_fp, wse_coarse_fp, 
                                      wsh_coarse_fp,dem_coarse_fp,
                              method, params, decay_method_d,
                              use_wsh,use_dem,pluvial_params,
                              tmpdir, logger, caseName):
    
 
    logger.info(f'caseName: {caseName}\n{pprint.pformat(decay_method_d)}\n{tmpdir}')
 
    from ..fdsc.main import downscale_wse_raster as func
    
    if not use_wsh:
        wsh_coarse_fp = None
        
    if not use_dem:
        dem_coarse_fp=None
    
    #downscale_pluvial_costGrow_xr()
    func(dem_fine_fp, wse_coarse_fp, 
         method=method,  
         out_dir=os.path.join(tmpdir, caseName),
         logger=logger,
         pluvial=True, 
         wsh_coarse_fp=wsh_coarse_fp,dem_coarse_fp=dem_coarse_fp,
         decay_method_d=decay_method_d,
         **params, **pluvial_params) 