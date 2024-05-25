'''
Created on May 25, 2024

@author: cef
'''


#===============================================================================
# IMPORTS------
#===============================================================================
import os, pathlib, pytest, logging, sys, tempfile

from fdsc.hp.dirz import recursive_file_search

#===============================================================================
# helpers
#===============================================================================



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
    test_data_lib = recursive_file_search(test_data_dir, ['.tif'])

#===============================================================================
# #add toy data
#===============================================================================
from tests.data_toy import test_data_lib as toy_test_data_lib
from tests.data_toy import ar_to_geoTiff

temp_dir = tempfile.mkdtemp()

#create raster files from toy data
"""would be nicer to do this on demand"""
for caseName,v in toy_test_data_lib.copy().items():
    toy_test_data_lib[caseName]['00_raw'] = dict() 
    for k in ['dem_fine', 'wse_coarse']:
        ar = v[k+'_ar']
        fp = os.path.join(temp_dir, 'conftest', caseName, k+'.tif')
        toy_test_data_lib[caseName]['00_raw'][k] = ar_to_geoTiff(ar, fp)
 
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
                )
    
    #get a new logger and lower it to avoid messing with dependencies
    log = logging.getLogger('test')
    log.setLevel(logging.DEBUG)
    
    
    return log


#===============================================================================
# FIXTURES.DATA---------
#===============================================================================
@pytest.fixture(scope='function')
def dem_fine_fp(caseName, phase):
    assert caseName in test_data_lib, f'unrecognized caseName: \'{caseName}\''
    return test_data_lib[caseName][phase]['dem_fine']

@pytest.fixture(scope='function')
def wse_coarse_fp(caseName, phase):
    assert caseName in test_data_lib, f'unrecognized caseName: \'{caseName}\''
    return test_data_lib[caseName][phase]['wse_coarse']

@pytest.fixture(scope='function')
def wse_fine_xr(caseName, phase):
    return test_data_lib[caseName][phase]['wse_fine_xr']






