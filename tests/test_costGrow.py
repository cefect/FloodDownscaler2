'''
Created on May 25, 2024

@author: cef
'''


import os, pathlib, pytest, logging, sys

import numpy.ma as ma
import xarray as xr

from fdsc.hp.xr import xr_to_GeoTiff, dataarray_from_masked
 
@pytest.mark.parametrize('phase', ['02_wp'])
@pytest.mark.parametrize('caseName',[
    #'case_01', #no DEM
    #'case_ahr',
    #'case_jordan',
    'case_toy1',
    ])
def test_cost_distance_fill(wse_fine_xr):
 
    from fdsc.alg.costGrow import _distance_fill_cost as func     
    func(wse_fine_xr.to_masked_array(), xr.ones_like(wse_fine_xr).data)



@pytest.mark.dev
@pytest.mark.parametrize('phase', ['02_wp'])
@pytest.mark.parametrize('caseName',[
    #'case_01', #no DEM
    #'case_ahr',
    #'case_jordan',
    #'case_toy1',
    'case_f3n2e100', #EPSG:4326. 9000x9000, 3:1
    ])
@pytest.mark.parametrize('_method, params',
                         [
                             #'distance_transform_bf', #MUCH slower
                             ('distance_transform_cdt', dict(metric='taxicab')),
                             ('distance_transform_cdt', dict(metric='chessboard')),
                             ('distance_transform_edt', dict()),
                             ])
def test_distance_fill(wse_fine_xr, _method, params, tmpdir):
    
    print(_method)
    print(params)
    
    from fdsc.alg.costGrow import _distance_fill as func
    result = func(wse_fine_xr.to_masked_array(), method=_method, **params)
    
    print(f'finished w/ {result.shape}')
    
    #output
    da = dataarray_from_masked(ma.MaskedArray(result), wse_fine_xr) 
    xr_to_GeoTiff(da, os.path.join(tmpdir, f'{_method}.tif'), compress=None) 
    
    