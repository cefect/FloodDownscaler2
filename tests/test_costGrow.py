'''
Created on May 25, 2024

@author: cef
'''

#===============================================================================
# import sys, os
# print(sys.executable)
# print(f'sys.version: {sys.version}')
# print(f'os.getcwd(): {os.getcwd()}')
# print('sys.path\n' + '\n'.join(sys.path))
# print('PYTHONPATH')
# if 'PYTHONPATH' in os.environ: print('\n'.join(os.environ['PYTHONPATH'].split(';')))
#===============================================================================


import os, pathlib, pytest, logging, sys

import numpy.ma as ma
import xarray as xr

from ..fdsc.hp.xr import xr_to_GeoTiff, dataarray_from_masked


 
 
@pytest.mark.parametrize('phase', ['02_wp'])
@pytest.mark.parametrize('caseName',[
    #'case_01', #no DEM
    'case_ahr',
    'case_jordan',
    'case_toy1',

    ])
def test_distance_fill_cost_wbt(wse_fine_xr, logger, tmpdir):
 
    from ..fdsc.alg.costGrow import _distance_fill_cost_wbt as func     
    func(wse_fine_xr, xr.ones_like(wse_fine_xr), log=logger, out_dir=tmpdir)



 
@pytest.mark.parametrize('phase', ['02_wp'])
@pytest.mark.parametrize('caseName',[
    #'case_01', #no DEM
    #'case_ahr',
    'case_jordan',
    'case_toy1',
    #'case_f3n2e100', #EPSG:4326. 9000x9000, 3:1
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
    
    from ..fdsc.alg.costGrow import _distance_fill as func
    result = func(wse_fine_xr.to_masked_array(), method=_method, **params)
    
    print(f'finished w/ {result.shape}')
    
    #output
    da = dataarray_from_masked(ma.MaskedArray(result), wse_fine_xr) 
    xr_to_GeoTiff(da, os.path.join(tmpdir, f'{_method}.tif'), compress=None)


 
@pytest.mark.parametrize('phase', ['02_wp'])
@pytest.mark.parametrize('caseName',[
    #'case_01', #no DEM
    #'case_ahr',
    'case_jordan',
    'case_toy1',
    #'case_f3n2e100', #EPSG:4326. 9000x9000, 3:1
    ]) 
@pytest.mark.parametrize('cd_backend', ['wbt'])
def test_distance_fill_cost_terrain(wse_fine_xr,dem_fine_xr, wse_coarse_xr, 
                                    tmpdir, logger, cd_backend):
 
    from ..fdsc.alg.costGrow import _distance_fill_cost_terrain as func
    result = func(wse_fine_xr, dem_fine_xr, wse_coarse_xr, 
                  log=logger, cd_backend=cd_backend, out_dir=tmpdir)
    
    print(f'finished w/ {result.shape}')
    
    #output
    da = dataarray_from_masked(ma.MaskedArray(result), wse_fine_xr) 
    xr_to_GeoTiff(da, os.path.join(tmpdir, f'{cd_backend}.tif'), compress=None)
    

@pytest.mark.dev
#@pytest.mark.parametrize('phase', ['02_wp'])
@pytest.mark.parametrize('caseName',[
    #'case_01', #no DEM
    #'case_ahr',
    'case_jordan',
    #'case_toy1',
    #'case_f3n2e100', #EPSG:4326. 9000x9000, 3:1
    ]) 
def test_xr_gdalslope(dem_fine_xr, logger, tmpdir):
    from ..fdsc.alg.costGrow import _xr_gdalslope as func
    
    da = func(dem_fine_xr, log=logger)
    
    xr_to_GeoTiff(da, os.path.join(tmpdir, f'xr_gdalslope.tif'), compress=None)
    
    
    
