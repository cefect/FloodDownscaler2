'''
Created on May 25, 2024

@author: cef
'''


#===============================================================================
# IMPORTS------
#===============================================================================
import os, pathlib, pytest, logging, sys, tempfile, pickle, copy
import rioxarray
from ..hp.dirz import recursive_file_search

#===============================================================================
# helpers
#===============================================================================

def _geoTiff_to_xr(fp): 
    return rioxarray.open_rasterio(fp,masked=False).squeeze().rio.write_nodata(-9999)

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
#load the file structure from the test data directory
from definitions import test_data_dir

if os.path.exists(test_data_dir):
    test_data_lib = recursive_file_search(test_data_dir, ['.tif', '.pkl'])

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
    for k in ['dem_fine', 'wse_coarse']:
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


@pytest.fixture(scope='session')
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
    log = logging.getLogger('t')
    log.setLevel(logging.DEBUG)
    
    
    return log


#===============================================================================
# FIXTURES.DATA---------
#===============================================================================
@pytest.fixture(scope='function')
def dem_fine_fp(caseName, phase):
    assert caseName in test_data_lib, f'unrecognized caseName: \'{caseName}\''
    return copy.deepcopy(test_data_lib[caseName][phase]['dem_fine'])

@pytest.fixture(scope='function')
def wse_coarse_fp(caseName, phase):
    assert caseName in test_data_lib, f'unrecognized caseName: \'{caseName}\''
    return copy.deepcopy(test_data_lib[caseName][phase]['wse_coarse'])


def _get_xr(caseName, phase, dataName):
    d = test_data_lib[caseName][phase]
    """
    print(test_data_lib[caseName][phase].keys())
    """
    #load from file once
    if not dataName in d:
        
        print(f'loading \'{dataName}\' from pickle file')
        
        k2 = dataName.replace('_xr', '')
        
        if not k2 in d:
            raise KeyError(k2)
        
        fp = d[k2]
        
        with open(fp, "rb") as f:
            d[dataName] = pickle.load(f)
            #return a copy
    return copy.deepcopy(test_data_lib[caseName][phase][dataName])

@pytest.fixture(scope='function')
def wse_fine_xr(caseName, phase):
    return _get_xr(caseName, phase, 'wse_fine_xr')

@pytest.fixture(scope='function')
def dem_fine_xr(caseName):
    return _get_xr(caseName, '00_inputs', 'dem_fine_xr')

@pytest.fixture(scope='function')
def wse_coarse_xr(caseName):
    return _get_xr(caseName, '00_inputs', 'wse_coarse_xr')









 



