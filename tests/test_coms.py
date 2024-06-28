'''
Created on Jun. 28, 2024

@author: cef
'''
import os, pathlib, pytest, logging, sys


@pytest.mark.parametrize('fine_dims', [(4000, 4000)])
@pytest.mark.parametrize('coarse_dims', [(72, 100), (201, 201)])
def test_distance_fill_cost_wbt(fine_dims, coarse_dims):
 
    from ..fdsc.coms import round_to_closest_shape as func     
    func(fine_dims, coarse_dims)