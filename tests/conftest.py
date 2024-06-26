'''
Created on Dec. 4, 2022

@author: cefect
'''
import pytest, os, tempfile, datetime
from collections import OrderedDict
import numpy as np
import numpy.ma as ma
import pandas as pd
import rasterio as rio
import shapely.geometry as sgeo
from shapely.geometry import mapping, Polygon
 
import fiona
import fiona.crs
from pyproj.crs import CRS

#project
from parameters import src_dir
from tests._params import epsg, bounds
from fdsc.hp._params import temp_dir
#from fdsc.base import nicknames_d

#helpers
#===============================================================================
# from hp.tests.tools.rasters import get_rlay_fp
# from hp.tests.conftest import init_kwargs
from fdsc.hp.logr import get_new_console_logger, logging
# from hp.rio import write_array, write_array2
# from hp.riom import assert_masked_ar
#===============================================================================


    
#===============================================================================
# TEST PARAMETERS-----------
#===============================================================================
# common test methodName and associated parameters.
"""because we use non-default pars for speeding up the tests
use:
    @pytest.mark.parametrize(*par_algoMethodKwargs)
"""
par_method_kwargs = {
    'CostGrow': {},
    'CostGrow': dict(loss_frac=0.01),
    'Basic': {},
    'SimpleFilter': {},
    'BufferGrowLoop': {'loop_range': range(2)},
    'Schumann14': {'buffer_size': float(2/3)}
}

par_algoMethodKwargs = ('method, kwargs', [(k,v) for k,v in par_method_kwargs.items()])
#@pytest.mark.parametrize(*par_algoMethodKwargs)
 


#check it
#miss_s = {e[0] for e in par_algoMethodKwargs[1]}.difference(nicknames_d.keys())
#assert miss_s==set(), 'parameter key mismatch: %s'%miss_s

#===============================================================================
# TEST DATA---------
#===============================================================================

crs_default = CRS.from_user_input(epsg)
bbox_default = sgeo.box(*bounds)

#not sure what happen to the fred01 data
#===============================================================================
# proj_lib = dict()
# proj_lib['fred01'] = {
#     # test raw data
#     'wse2_rlay_fp':os.path.join(src_dir, r'tests/data/fred01/testr_test00_0806_fwse.tif'),
#     'aoi_fp':os.path.join(src_dir, r'tests/data/fred01/aoi_T01.geojson'),
#     'crs':CRS.from_user_input(3979),
#     
#     # p1_downscale_wetPartials
#     'wse1_rlay2_fp':os.path.join(src_dir, r'tests/data/fred01/wse1_ar2.tif'),
#     
#     # p2_dp_costGrowSimple._filter_dem_violators
#     'wse1_rlay3_fp':os.path.join(src_dir, r'tests/data/fred01/wse1_ar3.tif'),
#         
#     'dem1_rlay_fp':os.path.join(src_dir, r'tests\data\fred01\dem.tif'),
#     
#     # validation data
#     'wse1_rlayV_fp':os.path.join(src_dir, r'tests/data/fred01/vali/wse1_arV.tif'),
#     'sample_pts_fp':os.path.join(src_dir, r'tests/data/fred01/vali/sample_pts_0109.geojson'),
#     'samp_gdf_fp':os.path.join(src_dir, r'tests/data/fred01/vali/samps_gdf_0109.pkl'),
#     'inun_vlay_fp':os.path.join(src_dir, r'tests/data/fred01/vali/inun_vali1.geojson'),
#     'hwm_pts_fp':os.path.join(src_dir, r'tests/data/fred01/vali/hwm_pts_0303.geojson'),
#     
#     # post data
#     'valiM_fp_d':{
#         'cgs':os.path.join(src_dir, r'tests/data/fred01/post/cgs_0109_valiMetrics.pkl'),
#         'noDP':os.path.join(src_dir, r'tests/data/fred01/post/none_0109_valiMetrics.pkl'),
#         },
#     }
#===============================================================================

#===============================================================================
# helpers-----
#===============================================================================


def get_xy_coords(transform, shape):
    """return an array of spatial values for x and y
    
    surprised there is no builtin
    
    this is needed  by xarray
    
    print(f'x, cols:{s[1]}    y, rows:{s[0]}')
    """
    transformer = rio.transform.AffineTransformer(transform) 
    x_ar, _ = transformer.xy(np.full(shape[1], 0), np.arange(shape[1]))  # rows, cols            
    _, y_ar = transformer.xy(np.arange(shape[0]), np.full(shape[0], 0))  # rows, cols
    
    return x_ar, y_ar
                      
 
 


def get_aoi_fp(bbox, crs=crs_default, ofp=None):
    
    if ofp is None:
        ofp = os.path.join(temp_dir, 'aoi.geojson')
        
    # write a vectorlayer from a single bounding box
    assert isinstance(bbox, Polygon)
    with fiona.open(ofp, 'w', driver='GeoJSON',
        crs=CRS.from_epsg(crs.to_epsg()),
        schema={'geometry': 'Polygon',
                'properties': {'id':'int'},
            },
 
        ) as c:
        
        c.write({ 
            'geometry':mapping(bbox),
            'properties':{'id':0},
            })
        
    return ofp

    
#===============================================================================
# MISC----
#===============================================================================
@pytest.fixture(scope='session')
def write():
    write = False
    if write:
        print('WARNING!!! runnig in write mode')
    return write


@pytest.fixture(scope='function')
def test_name(request):
    return request.node.name.replace('[', '_').replace(']', '_')


@pytest.fixture(scope='session')
def logger():
    return get_new_console_logger(level=logging.DEBUG)

@pytest.fixture(scope='function')
def init_kwargs(tmp_path,logger, test_name):
    return dict(
        out_dir=tmp_path, 
        tmp_dir=os.path.join(tmp_path, 'tmp_dir'),
        base_dir=tmp_path,
        #prec=prec,
        #proj_name='test', #probably a better way to propagate through this key 
        run_name=test_name[:8].replace('_',''),
        
        relative=True, 
        
        logger=logger, overwrite=True, logfile_duplicate=False,
        )


 
    
