'''
Created on May 25, 2024

@author: cef
'''
import os


def recursive_file_search(root_dir, search_extension_l):
    """
    Recursively searches a directory for files matching the given extension.
    Returns a nested dictionary mirroring the directory structure, containing
    only the matching files.

    Args:
        root_dir (str): The starting directory for the search.
        search_extension (str): The file extension to match (e.g., ".txt", ".pdf").

    Returns:
        dict: A nested dictionary representing the directory structure, with
              keys as folder names and values as either file names (if they
              match the extension) or further nested dictionaries (for
              subdirectories).
    """
    assert os.path.exists(root_dir), root_dir
    assert isinstance(search_extension_l, list)
    
    result = {}

    for item in os.listdir(root_dir):
        item_path = os.path.join(root_dir, item)

        if os.path.isdir(item_path):
            subdir_result = recursive_file_search(item_path, search_extension_l)
            if subdir_result:  # Only include subdirectories with matching files
                result[item] = subdir_result
        else:
            fn, ext = os.path.splitext(item)
            if ext in search_extension_l:
                result[fn] = item_path  # Store full path of matching file

    return result

#===============================================================================
# def get_filepaths_from_structure(test_data_dir):
#     """load data from directory structure"""
#  
#     raster_dict = {}
#  
#     for test_case_name in  os.listdir(test_data_dir):
#         
#     
#         # Find raster files within the subdirectory
#         sub_dir = os.path.join(test_data_dir, test_case_name)
#         if '.' in sub_dir: continue
#         raster_dict[test_case_name]=dict()
#         
#         for resolution in os.listdir(sub_dir):
#             if os.path.isfile(os.path.join(sub_dir, resolution)): continue
#             raster_dict[test_case_name][resolution] = dict()
#             
#             for fn in os.listdir(os.path.join(sub_dir, resolution)):
#                 fp=  os.path.join(sub_dir, resolution, fn)
#                 k=None
#                 if fn.endswith(".tif"):
#                     if fn.startswith("DEM_raw"):
#                         k='dem'
#                     elif fn.startswith("res_fluv"):
#                         k='wsh'
#                         
#                 if not k is None:
#                     raster_dict[test_case_name][resolution][k] = fp
#                             
#          
# 
#     return raster_dict
#===============================================================================