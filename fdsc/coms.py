'''
Created on May 25, 2024

@author: cef
'''

import warnings, math
from .assertions import *

def shape_ratio(da1, da2):
    """
    Calculates the ratio of shapes between two xarray DataArrays.

    Args:
        da1 (xr.DataArray): The first DataArray.
        da2 (xr.DataArray): The second DataArray.

    Returns:
        tuple: A tuple containing the ratios of corresponding dimensions 
               (da1_dim_size / da2_dim_size) for each shared dimension.
               If the DataArrays have different dimension names or number of dimensions, 
               raises a ValueError.
    """
    
    # Check if dimensions match
    if da1.dims != da2.dims:
        raise ValueError("DataArrays have different dimensions.")
    
    # Calculate ratios and return as a tuple
    ratios = tuple(da1.sizes[dim] / da2.sizes[dim] for dim in da1.dims)
    return ratios


def find_nearest_divisor(X, Y):
    # Find all factors of X
    factors = [i for i in range(1, X + 1) if X % i == 0]
    
    # Find the factor closest to Y
    closest_factor = min(factors, key=lambda k: abs(k - Y))
    
    return closest_factor

 


# Function to find the closest factors for coarse_shape dimensions
def round_to_closest_shape(fine_dims, coarse_dims):
    """not sure how robust this is"""
    new_dims = []
    for fine_dim, coarse_dim in zip(fine_dims, coarse_dims): 
        new_dims.append(find_nearest_divisor(fine_dim, coarse_dim))
    
    #check
    for fine_dim, coarse_dim in zip(fine_dims, new_dims): 
        ratio = fine_dim / coarse_dim
        if not np.isclose(ratio, int(ratio)):
            raise ValueError(f'Result is not an integer\n    {ratio}\n    new_dims={new_dims}')
        
        
    return tuple(new_dims)

 


def set_da_layerNames(da_layerName_d):
    """set the layer name attributes and check some expectations"""
    for expected_name, data_array  in da_layerName_d.items():
        
        #check existing layername
        if 'layerName' in data_array.attrs:
            actual_name = data_array.attrs['layerName']
            if actual_name != expected_name:
                warnings.warn(f"Unexpected layerName for {data_array.name}: Expected '{expected_name}', found '{actual_name}'")
        
        #set    
        data_array.attrs['layerName'] = expected_name
        
        #check geoTiff like
        assert_xr_geoTiff(data_array)