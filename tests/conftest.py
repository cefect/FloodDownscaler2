'''
Created on May 25, 2024

@author: cef
'''


#===============================================================================
# IMPORTS------
#===============================================================================
import os, pathlib, pytest, logging, sys, tempfile, pickle, copy, warnings
import rioxarray
import numpy as np
from ..fdsc.hp.dirz import recursive_file_search
from ..fdsc.hp.xr import coarsen_dataarray, resample_match_xr, wse_to_wsh_xr, xr_to_GeoTiff

#===============================================================================
# Parameters-------
#===============================================================================
write_to_test_data=True #control for writing test results to test_data_dir
if write_to_test_data:
    warnings.warn(f'write_to_test_data=True. test data will be over-written')

#===============================================================================
# helpers---------
#===============================================================================

def _geoTiff_to_xr(fp): 
    da = rioxarray.open_rasterio(fp,masked=False).squeeze().compute().rio.write_nodata(-9999)
    return da.where(da!=-9999, np.nan)

def _get_xr(caseName, phase, dataName):
 
    assert phase in test_data_lib[caseName], f'missing {caseName}.{phase}'
    d = test_data_lib[caseName][phase]
    """
    print(test_data_lib[caseName][phase].keys())
    """
    
    """switched to always load to workaorund deepcopy
    #load from file once
    if not dataName in d:"""
        
    k2 = dataName.replace('_xr', '')
    
    if not k2 in d:
        raise KeyError(k2)
    
    fp = d[k2]
    
    print(f'loading \'{dataName}\' from pickle file\n    {fp}')
    with open(fp, "rb") as f:
        da = pickle.load(f)
 
    da = da.rio.write_nodata(-9999)    
        
    """deepcopy is not copying the rio data
    #check it
    da = copy.deepcopy(test_data_lib[caseName][phase][dataName])"""
    from ..fdsc.assertions import assert_xr_geoTiff        
    assert_xr_geoTiff(da, msg='%s.%s.%s'%(caseName, phase, dataName))
    
        
    return da


#===============================================================================
# TEST_DATA------------
#===============================================================================
"""the test data library is kept separate (should make available for download)
contains these cases:
    case_01\
    case_ahr\
    case_jordan\
    
additional 'phase' parameters are used to allow for intermediate test data
    
"""
#===============================================================================
# load from directory
#===============================================================================
from definitions import test_data_dir

if os.path.exists(test_data_dir):
    test_data_lib = recursive_file_search(test_data_dir, ['.tif', '.pkl', '.gpkg'])
else:
    raise IOError(f'no test data directory found at: {test_data_dir}')

#===============================================================================
# #add toy data
#===============================================================================
from .data_toy import test_data_lib as toy_test_data_lib
from .data_toy import ar_to_geoTiff

temp_dir = tempfile.mkdtemp()

#create raster files from toy data
"""would be nicer to do this on demand"""
for caseName,v in toy_test_data_lib.copy().items():
    toy_test_data_lib[caseName]['00_raw'] = dict() 
    for k in ['dem_fine', 'wse_coarse', 'dem_coarse']:
        ar = v[k+'_ar']
        fp = os.path.join(temp_dir, 'conftest', caseName, k+'.tif')
        toy_test_data_lib[caseName]['00_raw'][k] = ar_to_geoTiff(ar, fp)
    
    #handle toy data w/ some data files
    if caseName in test_data_lib:
        toy_test_data_lib[caseName].update(test_data_lib[caseName])
 
#update the test lib
test_data_lib.update(toy_test_data_lib)



#===============================================================================
# fixtures---------
#===============================================================================


@pytest.fixture(scope='function')
def logger():
    logging.basicConfig(
                #filename='xCurve.log', #basicConfig can only do file or stream
                force=True, #overwrite root handlers
                stream=sys.stdout, #send to stdout (supports colors)
                level=logging.INFO, #lowest level to display
                format='%(asctime)s %(levelname)s %(name)s: %(message)s',  # Include timestamp
                datefmt='%H:%M:%S'  # Format for timestamp
                )
    
    #get a new logger and lower it to avoid messing with dependencies
    log = logging.getLogger(str(os.getpid()))
    log.setLevel(logging.DEBUG)
    
    
    return log


#===============================================================================
# FIXTURES.DATA---------
#===============================================================================
@pytest.fixture(scope='function')
def dem_fine_fp(caseName):
    phase = '00_raw'
    assert caseName in test_data_lib, f'unrecognized caseName: \'{caseName}\''
    assert phase in test_data_lib[caseName], f'missing phase: {caseName}.{phase}'
    return copy.deepcopy(test_data_lib[caseName][phase]['dem_fine'])

@pytest.fixture(scope='function')
def wse_coarse_fp(caseName, phase):
    assert caseName in test_data_lib, f'unrecognized caseName: \'{caseName}\''
    return copy.deepcopy(test_data_lib[caseName][phase]['wse_coarse'])


@pytest.fixture(scope='function')
def dem_coarse_fp(caseName, phase):
    assert caseName in test_data_lib, f'unrecognized caseName: \'{caseName}\''
    assert 'dem_coarse' in test_data_lib[caseName][phase], f'missing dem_coarse: {caseName}.{phase}'
    return copy.deepcopy(test_data_lib[caseName][phase]['dem_coarse'])



@pytest.fixture(scope='function')
def wse_fine_xr(caseName, phase):
    return _get_xr(caseName, phase, 'wse_fine_xr')

@pytest.fixture(scope='function')
def dem_fine_xr(caseName):
    return _get_xr(caseName, '00_inputs', 'dem_fine_xr')

@pytest.fixture(scope='function')
def wse_coarse_xr(caseName):
    return _get_xr(caseName, '00_inputs', 'wse_coarse_xr')




@pytest.fixture(scope='function')
def dem_coarse_xr(caseName, phase, dem_fine_xr, wse_coarse_xr):
    
    d = test_data_lib[caseName][phase]
    dataName = 'dem_coarse_xr'
    
    #===========================================================================
    # build
    #===========================================================================
#===============================================================================
#     if not dataName in d:    
#         from analysis.hp.xr import coarsen_dataarray
# 
#         d[dataName] = coarsen_dataarray(dem_fine_xr, wse_coarse_xr)
#  
#     return copy.deepcopy(test_data_lib[caseName][phase][dataName])
#===============================================================================
    

    #return coarsen_dataarray(dem_fine_xr, wse_coarse_xr)
    return resample_match_xr(dem_fine_xr, wse_coarse_xr)


@pytest.fixture(scope='function')
def wsh_coarse_fp(caseName, wse_coarse_fp, dem_coarse_xr, tmpdir, logger):
    """ most functions are setup to take WSH.. so we want this as a test endpoint"""
    assert caseName in test_data_lib, f'unrecognized caseName: \'{caseName}\''
    
    phase='00_raw'
    d = test_data_lib[caseName][phase]
 
    dataName = 'wsh_coarse'
    if not dataName in d:
        #===========================================================================
        # create WSH data
        #===========================================================================
     
        wse_xr = _geoTiff_to_xr(wse_coarse_fp) 
        
        #from FloodDownscaler2.fdsc.hp.xr import wse_to_wsh_xr, xr_to_GeoTiff
        
        wsh_xr = wse_to_wsh_xr(dem_coarse_xr, wse_xr, allow_low_wse=True, log=logger)
        
        #===========================================================================
        # write
        #===========================================================================
        
        d[dataName] = xr_to_GeoTiff(wsh_xr, os.path.join(tmpdir, 'wsh_coarse.tif'), compress=None)
        
    return test_data_lib[caseName][phase][dataName]





 



