'''
Created on May 25, 2024

@author: cef
'''
import os, tempfile


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

def get_od(out_dir):
    if out_dir is None:
        out_dir = tempfile.mkdtemp()
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)        
    return out_dir


def make_dir(path):
    """
    Create the directory for the given path if it does not exist.
    If the path is a file, create the parent directory.
    If the path is a directory, create the directory itself.
    
    Parameters:
    path (str): The file or directory path.
    """
    if os.path.splitext(path)[1]:  # Check if there is a file extension
        # If the path is a file, create the parent directory
        bdir = os.path.dirname(path)
        if not os.path.exists(bdir):
            os.makedirs(bdir)
    else:
        # If the path is a directory, create it if it does not exist
        if not os.path.exists(path):
            os.makedirs(path)

 