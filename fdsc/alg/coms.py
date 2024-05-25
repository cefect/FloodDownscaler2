'''
Created on May 25, 2024

@author: cef
'''


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