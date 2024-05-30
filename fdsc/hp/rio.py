'''
Created on May 26, 2024

@author: cef
'''

import rasterio, os, tempfile
from rasterio.warp import calculate_default_transform, reproject, Resampling
from pyproj import CRS, Transformer


#===============================================================================
# HELPERS------
#===============================================================================

def _get_od(out_dir):
    if out_dir is None:
        out_dir = tempfile.mkdtemp()
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        
    return out_dir
        
        

def geographic_to_projected(raster_fp, log=None, out_dir=None):
    """Reproject a geographic raster to a projected one using best UTM"""
    
    #===========================================================================
    # defaults
    #===========================================================================
    plog = lambda msg: print(msg) if log is None else log.debug(msg)
    
     
    
    
    
    with rasterio.open(raster_fp) as src:
        # Check if the raster is geographic, if not...do nothing
        if not src.crs.is_geographic:
            plog(f'non-geographic crs detected {src.crs}')
            return raster_fp
        
        #=======================================================================
        # setup
        #=======================================================================
        out_dir = _get_od(out_dir)
        old_crs = src.crs
        
 
        # Identify the appropriate UTM CRS using the center of the tile
        lon, lat = src.lnglat()
        utm_crs = CRS.from_epsg(32600 + int((lon + 180) / 6) % 60)
        
        # Calculate the transform and the new dimensions
        transform, width, height = calculate_default_transform(
            src.crs, utm_crs, src.width, src.height, *src.bounds)
        
        
        # Create a new raster to store the reprojected data
        ofp = os.path.join(out_dir, os.path.basename(raster_fp).replace('.tif', f'_{utm_crs.to_epsg()}.tif')) 
        
        
 
        with rasterio.open(ofp, 'w', driver='GTiff', 
                           height=height, width=width, count=src.count, 
                           dtype=src.dtypes[0], crs=utm_crs, transform=transform
                           ) as dst:
            
            # Reproject the raster using the identified UTM
            reproject(
                source=rasterio.band(src, 1),
                destination=rasterio.band(dst, 1),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=utm_crs,
                resampling=Resampling.nearest)
            
    plog(f'reprojected from {old_crs.to_epsg()} to {utm_crs.to_epsg()}\n    {ofp}')
    
    return ofp
            
            
            
def get_geotiff_shape(filepath):
    with rasterio.open(filepath) as src:
        return (src.height, src.width)  # Return as NumPy array
            
            
            
            
            
            
            
            
