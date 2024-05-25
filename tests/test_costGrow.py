'''
Created on May 25, 2024

@author: cef
'''


import os, pathlib, pytest, logging, sys

import xarray as xr


@pytest.mark.dev
@pytest.mark.parametrize('phase', ['02_wp'])
@pytest.mark.parametrize('caseName',[
    #'case_01', #no DEM
    #'case_ahr',
    #'case_jordan',
    'case_toy1',
    ])
def test_cost_distance_fill(wse_fine_xr):
    
 
    from fdsc.alg.costGrow import _cost_distance_fill as func
     
     
     
    func(wse_fine_xr.to_masked_array(), xr.ones_like(wse_fine_xr).data)
