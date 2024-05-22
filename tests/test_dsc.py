'''
Created on Dec. 4, 2022

@author: cefect

test downscaling methods


TODO: clean up fixtures for retriving/building test data
'''
#===============================================================================
# IMPORTS----------
#===============================================================================
import pytest, copy, os, random, re
import numpy as np
import pandas as pd

#import shapely.geometry as sgeo
 
xfail = pytest.mark.xfail

#from fdsc.scripts.disag import disag
#from fdsc.base import nicknames_d

from fdsc.control import Dsc_Session as Session
from fdsc.main import downscale
 
#from fdsc.bufferLoop import ar_buffer
from fdsc.hp.rasters import get_rlay_fp

from tests.conftest import (
     proj_lib,get_aoi_fp, 
    par_method_kwargs,temp_dir,
    par_algoMethodKwargs,
 
    )
 
#===============================================================================
# test data------
#===============================================================================
from tests.data.toy import (
    aoi_box, bbox_default, proj_ar_d, crs_default
    )

#build rasters
toy_d =dict()
for k, ar in proj_ar_d.items():
    toy_d[k] = get_rlay_fp(ar, k, out_dir=temp_dir, crs=crs_default, bbox=bbox_default)
    

 
toy_d['aoi'] = get_aoi_fp(aoi_box, crs=crs_default)


test_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'test_dsc')
assert os.path.exists(test_data_dir)

#===============================================================================
# fixtures------------
#===============================================================================
@pytest.fixture(scope='function')
def wrkr(init_kwargs, crs= crs_default):
    with Session(crs=crs, **init_kwargs) as session:
        yield session
 

#===============================================================================
# tests-------
#===============================================================================

def test_init(wrkr):
     
    pass
    


@pytest.mark.parametrize('dem_fp, wse_fp, aoi_fp', [
    (toy_d['dem1'], toy_d['wse2'], toy_d['aoi']),
    #(proj_lib['fred01']['dem1_rlay_fp'], proj_lib['fred01']['wse2_rlay_fp'], proj_lib['fred01']['aoi_fp'])
    ]) 
def test_p0_clip(dem_fp, wse_fp, aoi_fp, tmp_path, wrkr): 
    wrkr._set_aoi(aoi_fp)
    wrkr.p0_clip_rasters(dem_fp, wse_fp,  out_dir=tmp_path)
    
 

@pytest.mark.parametrize('dem_fp, wse_fp', [
    (toy_d['dem1'], toy_d['wse2']),
 
    ]) 
def test_p1(dem_fp, wse_fp, wrkr):    
    wrkr.p1_wetPartials(wse_fp, dem_fp)


tdir = lambda x: os.path.join(test_data_dir, 'test_04_isolated', x) 


@pytest.mark.parametrize('method, clump_cnt, wse_raw_fp, wse_fp',
                         [
                            ('area', 1, None, toy_d['wse13']),
                            ('area', 2, None, toy_d['wse13']),
                            ('pixel', None, toy_d['wse2'], toy_d['wse13']),
                            ('pixel', None, tdir('wse_r200_0121.tif'),tdir('wse_r005_0121.tif')),
                            ('pixel_polygon', None, tdir('wse_r200_0121.tif'),tdir('wse_r005_0121.tif')),
                             ('skimage_label', None, tdir('wse_r200_0121.tif'),tdir('wse_r005_0121.tif')),
                             ]
                         )
def test_04_isolated(wse_fp, wrkr, method, clump_cnt, wse_raw_fp):
    wrkr._04_isolated(wse_fp, method=method, clump_cnt=clump_cnt, wse_raw_fp=wse_raw_fp)
    

@pytest.mark.parametrize('dem_fp, wse_fp', [
    (toy_d['dem1'], toy_d['wse13']),
 
    ])
def test_p2_bufferGrow(dem_fp, wse_fp, wrkr):
    wrkr.get_bufferGrowLoop_DP(wse_fp, dem_fp, loop_range=range(3))
    

 


@pytest.mark.parametrize('dem_fp, wse_fp', [
    (toy_d['dem1'], toy_d['wse2']),
 
    ]) 
@pytest.mark.parametrize('backend', [
    #'gr', 
    'rio'])
def test_schu14(dem_fp, wse_fp, wrkr, backend):
    wrkr.run_schu14(wse_fp, dem_fp, buffer_size=float(2/3), r2p_backend=backend)
    


 

 
@pytest.mark.parametrize('dem_fp, wse_fp', [
    (toy_d['dem1'], toy_d['wse2']),
    #(proj_lib['fred01']['dem1_rlay_fp'], proj_lib['fred01']['toy_d['wse2']'])
    ])
@pytest.mark.parametrize('method_pars', [par_method_kwargs])
def test_run_dsc_multi(dem_fp, wse_fp, method_pars, wrkr):
    """
    FAILING: one of the metadata containers is out of compliance on CostGrow?
    """
    wrkr.run_dsc_multi(dem_fp, wse_fp, method_pars=method_pars)
    
    
tdir = lambda x: os.path.join(os.path.dirname(test_data_dir), 'jordan', x) 

@pytest.mark.dev
@pytest.mark.parametrize('dem_fp, wse_fp', [
    #(toy_d['dem1'], toy_d['wse2']),
    #(proj_lib['fred01']['dem1_rlay_fp'], proj_lib['fred01']['toy_d['wse2']'])
    (tdir('dem_fine_0522.tif'), tdir('wse_coarse_0522.tif')),
    ])
@pytest.mark.parametrize(*par_algoMethodKwargs)
def test_downscale(dem_fp, wse_fp, method, kwargs, tmp_path):
    assert os.path.exists(dem_fp)
    downscale(dem_fp, wse_fp, method=method, out_dir=tmp_path, **kwargs)
 
    
    
