'''
Created on Mar. 2, 2023

@author: cefect

cost grow downscale algorthim
'''


import os, datetime, shutil
import numpy as np
import numpy.ma as ma
import pandas as pd
import rasterio as rio
from rasterio import shutil as rshutil

import geopandas as gpd


#helpers
from hp.basic import now
from hp.gdal import getNoDataCount
from hp.rio import (
 
    write_array, assert_spatial_equal, load_array, 
      get_ds_attr, 
    )

from hp.riom import write_extract_mask, write_array_mask, assert_mask_fp
from hp.pd import view
from fdsc.base import assert_type_fp

#project
from fdsc.simple import WetPartials

from fdsc.base import (
    assert_dem_ar, assert_wse_ar, rlay_extract, assert_partial_wet
    )

class CostGrow(WetPartials):
    
    def __init__(self,
                 run_dsc_handle_d=None, 
                 **kwargs):
        
        if run_dsc_handle_d is None: run_dsc_handle_d=dict()
        
        run_dsc_handle_d['CostGrow'] = self.run_costGrow #add your main method to the caller dict
        
        super().__init__(run_dsc_handle_d=run_dsc_handle_d, **kwargs)
        
    def run_costGrow(self,wse_fp=None, dem_fp=None, 
                     cost_fric_fp=None,
                     clump_cnt=1,
                     clump_method='pixel',
                     loss_frac=0.0,
                              **kwargs):
        """run CostGrow pipeline
        
        Params
        -----------
        wse_fp: str
            coarse WSE to downsample
            
        """
        method='CostGrow'
        log, tmp_dir, out_dir, ofp, resname = self._func_setup(self.nicknames_d[method],  **kwargs)
        skwargs = dict(logger=log, out_dir=tmp_dir, tmp_dir=tmp_dir)
        meta_lib=dict()
        assert_type_fp(wse_fp, 'WSE')
        assert_type_fp(dem_fp, 'DEM')
        
        #skwargs, meta_lib, log, ofp, start = self._func_setup_dsc(nicknames_d[method], wse_fp, dem_fp, **kwargs)
        downscale = self.downscale
        self._set_profile(dem_fp) #set raster profile
        #=======================================================================
        # p1: wet partials
        #=======================================================================                
        wse1_wp_fp, meta_lib['p1_wp'] = self.p1_wetPartials(wse_fp, dem_fp, downscale=downscale,**skwargs)
  
        #=======================================================================
        # p2: dry partials
        #=======================================================================
        wse1_dp_fp, meta_lib['p2_DP'] = self.p2_costGrow_dp(wse1_wp_fp, dem_fp,ofp=ofp, 
                                                             cost_fric_fp=cost_fric_fp,
                                                             clump_kwargs=dict(
                                                                 clump_cnt=clump_cnt,
                                                                 method=clump_method,
                                                                 wse_raw_fp=wse_fp,
                                                                 ),
                                                             decay_kwargs=dict(
                                                                 wse_raw_fp=wse1_wp_fp,
                                                                 loss_frac=loss_frac,
                                                                 ),
                                                              **skwargs)     
 
        
        return wse1_dp_fp, meta_lib
        
        
        
    
    def p2_costGrow_dp(self, wse_fp, dem_fp,
                        cost_fric_fp=None,
                        clump_kwargs=dict(),
                        decay_kwargs=dict(),
                         **kwargs):
        """treat dry partials with costGrow"""
        #=======================================================================
        # defaults
        #=======================================================================
        log, tmp_dir, out_dir, ofp, resname = self._func_setup('cg_dp', subdir=False, **kwargs)
        skwargs = dict(logger=log, out_dir=tmp_dir, tmp_dir=tmp_dir)
        meta_lib={'smry':{'wse_fp':wse_fp}}
        start=now()
        assert_type_fp(wse_fp, 'WSE')
        self._set_profile(dem_fp) #set profile for session raster writing
        
        #=======================================================================
        # grow/buffer out the WSE values
        #=======================================================================
        costAlloc_fp, meta_lib['costDistanceGrow'] = self._01_grow(
            wse_fp, cost_fric_fp=cost_fric_fp, **skwargs)
        
        #=======================================================================
        # decay
        #=======================================================================
        decay_fp, meta_lib['decay_distance'] = self._02_decay(costAlloc_fp, **decay_kwargs, **skwargs)
 
        #=======================================================================
        # stamp out DEM violators
        #=======================================================================
        wse1_ar1_fp, meta_lib['filter_dem'] = self._filter_dem_violators(dem_fp, decay_fp, **skwargs)
        
        #report
        if __debug__:
            og_noDataCount = getNoDataCount(wse_fp)
            new_noDataCount = meta_lib['filter_dem']['violation_count']
            assert og_noDataCount>0            
            
            assert   new_noDataCount<og_noDataCount
            
            log.info(f'dryPartial growth from {og_noDataCount} to {new_noDataCount} nulls '+\
                     f'({new_noDataCount/og_noDataCount:.2f})')
        
        #=======================================================================
        # remove isolated 
        #======================================================================= 
        wse1_ar2_fp, meta_lib['filter_iso'] = self._03_isolated(wse1_ar1_fp,
                                                                    **clump_kwargs,
                                                                     **skwargs)
        
        assert_spatial_equal(wse_fp, wse1_ar2_fp)
        #=======================================================================
        # wrap
        #=======================================================================
        rshutil.copy(wse1_ar2_fp, ofp)
        tdelta = (now()-start).total_seconds()
        meta_lib['smry']['tdelta'] = tdelta
        log.info(f'finished in {tdelta:.2f} secs')
        
        return ofp, meta_lib
    
    def _filter_dem_violators(self, dem_fp, wse_fp, **kwargs):
        """replace WSe values with nodata where they dont exceed the DEM"""
        #=======================================================================
        # defautls
        #=======================================================================
        log, tmp_dir, out_dir, ofp, resname = self._func_setup('dem_filter', subdir=True,  **kwargs)
        assert_spatial_equal(dem_fp, wse_fp)
        
        """no... often we pass a costDistance raster which is WSE-like, but has no nulls
        assert_type_fp(wse_fp, 'WSE')"""
        #=======================================================================
        # load arrays
        #=======================================================================
        with rio.open( #load arrays
            wse_fp, mode='r') as ds:
            wse_ar = ds.read(1)
            assert not np.isnan(wse_ar).any(), 'shouldnt have any  nulls (we filled it!)'
            
        with rio.open(dem_fp, mode='r') as ds:
            dem1_ar = ds.read(1)
            
        #=======================================================================
        # #array math
        #=======================================================================
        bx_ar = wse_ar <= dem1_ar
        assert_partial_wet(bx_ar, msg='wse_ar <= dem1_ar')
 
        wse1_mar1 = ma.array(
            np.where(np.invert(bx_ar), wse_ar, np.nan),
            mask=bx_ar, fill_value=-9999)
        
        log.info(f'filtered {bx_ar.sum()}/{bx_ar.size} wse values which dont exceed the DEM')
        #=======================================================================
        # #dump to raster
        #=======================================================================
        #rlay_kwargs = get_write_kwargs(dem_fp, driver='GTiff', masked=False)
        wse1_ar1_fp = self.write_array(wse1_mar1, resname='wse1_ar3', 
                                       out_dir=out_dir,  logger=log, ofp=ofp) 
        
        
        #=======================================================================
        # meta
        #=======================================================================
        assert_type_fp(wse1_ar1_fp, 'WSE')
        
        meta_d={'size':wse_ar.size, 'wse1_ar1_fp':wse1_ar1_fp}
        if __debug__:
            meta_d['violation_count'] = bx_ar.astype(int).sum()
        
        
        return wse1_ar1_fp, meta_d
    
    def _01_grow(self, wse_fp,
                                 cost_fric_fp=None,
                                 
                                 **kwargs):
        """cost grow/allocation using WBT
        
        Params
        ----------
        cost_fric_fp: str (optional)
            filepath to cost friction raster
            if None: netural cost is used
        """
        start = now()
        log, tmp_dir, out_dir, ofp, resname = self._func_setup('01grow', subdir=True,  **kwargs)
        log.info(f'on {wse_fp}')
        meta_d = dict()
        #=======================================================================
        # costDistance
        #=======================================================================
        #fillnodata in wse (for source)
        wse_fp1 = os.path.join(tmp_dir, f'wse1_fnd.tif')
        assert self.convert_nodata_to_zero(wse_fp, wse_fp1) == 0
        
        #build cost friction (constant)\
        if cost_fric_fp is None:
            cost_fric_fp = os.path.join(tmp_dir, f'cost_fric.tif')
            assert self.new_raster_from_base(wse_fp, cost_fric_fp, value=1.0, data_type='float') == 0
        meta_d['costFric_fp'] = cost_fric_fp
        
        #compute backlink raster
        backlink_fp = os.path.join(out_dir, f'backlink.tif')
        assert self.cost_distance(wse_fp1, 
            cost_fric_fp, 
            os.path.join(tmp_dir, f'backlink.tif'), backlink_fp) == 0
        
        meta_d['backlink_fp'] = backlink_fp
            
        log.info(f'built costDistance backlink raster \n    {backlink_fp}')
        
        #=======================================================================
        # costAllocation
        #=======================================================================
        costAlloc_fp = os.path.join(out_dir, 'costAllocation.tif')
        assert self.cost_allocation(wse_fp1, backlink_fp, costAlloc_fp) == 0
        meta_d['costAlloc_fp'] = costAlloc_fp
        
        #=======================================================================
        # wrap
        #=======================================================================
        
        
        assert_spatial_equal(costAlloc_fp, wse_fp)
        assert_type_fp(wse_fp, 'WSE')
        
        tdelta = (now() - start).total_seconds()
        meta_d['tdelta'] = tdelta
        log.info(f'finished in {tdelta}\n    {costAlloc_fp}')
        return costAlloc_fp, meta_d
    
 
    
    def _02_decay(self,
                        costAlloc_fp, 
                        wse_raw_fp=None,
                        loss_frac=0.0,
                        **kwargs):
        """add some decay to the grow values
        
        
        Params
        -------
        wse1_wp_fp: str
            filepath to wet-partial treated downscaled WSE
            
        loss_frac: float
            value multiiplied  by distance to obtain the loss value
            this is equivalent to the minimum expected water surface slope
            
        """
        

        
        
        log, tmp_dir, out_dir, ofp, resname = self._func_setup('02decay', subdir=True,  **kwargs)
        
        if loss_frac==0.0:
            log.warning(f'got loss_frac=0.0.... skipping decay')
            return costAlloc_fp, dict()
        
        meta_d=dict()
        
        assert_spatial_equal(costAlloc_fp, wse_raw_fp)
        #=======================================================================
        # prep wse mask
        #=======================================================================
        with rio.open(wse_raw_fp, mode='r') as ds: 
            profile = ds.profile
            mar = ds.read(1, masked=True)
            
        wse_raw_mask_fp = os.path.join(tmp_dir, 'wse_raw_mask.tif')
        with rio.open(wse_raw_mask_fp, 'w', **profile) as ds:
            ds.write(np.where(mar.mask, 0.0, 1.0), indexes=1)
        
        log.debug(f'wrote wse_raw mask to \n    {wse_raw_mask_fp}')
        
        #=======================================================================
        # compute distance from mask
        #=======================================================================
        dist_fp = os.path.join(tmp_dir, 'euclidean_distance.tif')
        #horizontal distance (not pixels)
        assert self.euclidean_distance(wse_raw_mask_fp, dist_fp)==0
        
        meta_d.update({'wse_raw_mask_fp':wse_raw_mask_fp, 'dist_fp':dist_fp})
        #=======================================================================
        # distance-based decay
        #=======================================================================
        decay_fp = os.path.join(tmp_dir, 'decay.tif')
        """still not working
        assert self.raster_calculator(decay_fp, statement=f'\'{dist_fp}\'*{loss_frac}')==0
        """
        with rio.open(dist_fp, mode='r') as ds:
            mar = ds.read(1, masked=True)
            
        decay_mar = mar*loss_frac
        
        #preserving mask... but shouldnt matter... will apply this as subtraction
        with rio.open(decay_fp, 'w', **profile) as ds:            
            ds.write(decay_mar, indexes=1)
            
        log.debug(f'built decay raster \n    {decay_fp}')
        
        #=======================================================================
        # apply decay
        #=======================================================================
        #load the cost alloc raster
        with rio.open(costAlloc_fp, 'r') as ds:
            wse_mar = ds.read(1, masked=True)
        
        #check masks are equal
        """not sure why this fails...
        shouldnt matter
        assert np.array_equal(wse_mar.mask, decay_mar.mask)"""
            
        
        #write the subtraction
        costAlloc_decay_fp = os.path.join(out_dir, 'wse_decay.tif')
        with rio.open(costAlloc_decay_fp, 'w', **profile) as ds:
            
            #subtract and use original mask
            wse_decay_mar = ma.array(
                wse_mar.data-decay_mar.data,
                mask=wse_mar.mask, fill_value=-9999)
            
            #check
            """this is ok actually... will be filtered by teh DEM anyways
            assert wse_decay_mar.data.min()>0.0, f'got negatives'"""
            
                        
            ds.write(wse_decay_mar, indexes=1)
        
        #=======================================================================
        # wrap
        #=======================================================================
        meta_d.update({'decay_fp':decay_fp, 'costAlloc_decay_fp':costAlloc_decay_fp})
        log.debug(f'applied decay to costAlloc raster')
        
        return costAlloc_decay_fp, meta_d

    def _03_isolated(self, wse_fp, clump_cnt=1,
                         method='area',
                         min_pixel_frac=0.01,
                         wse_raw_fp=None,
                         **kwargs):
        """remove isolated cells from grid using WBT
        
        
        Params
        -------
        wse_fp: str
            filepath to fine resolution WSE on which to apply the isolated filter
            
        clump_cnt: int
            number of clumps to select
        
        method: str
            method for selecting isolated flood groups
            'area': take the n=clump_cnt largest areas (fast)
            'pixel': use polygons and points to select the groups of interest
            
            
        min_pixel_frac: float
            for method='pixel', the minimum fraction of total domain pixels allowed for the clump search
            
        wse_raw_fp: str
            for method='pixel', raw WSE from which to get intersect points
            
        """
        log, tmp_dir, out_dir, ofp, resname = self._func_setup('03isolated', subdir=True,  **kwargs)
        start = now()
        meta_d=dict()
        assert get_ds_attr(wse_fp, 'nodata')==-9999
        assert_type_fp(wse_fp, 'WSE', msg='filter_iso input')
        #=======================================================================
        # #convert to mask
        #=======================================================================
        """wbt.clump needs 1s and 0s"""
        mask_fp = write_extract_mask(wse_fp, out_dir=out_dir, maskType='native')
        assert_mask_fp(mask_fp,  maskType='native')
        #=======================================================================
        # #clump it
        #=======================================================================
        clump_fp = os.path.join(tmp_dir, 'clump.tif')
        assert self.clump(mask_fp, clump_fp, diag=False, zero_back=True)==0
        meta_d['clump_fp'] = clump_fp
        meta_d['clump_mask_fp'] = mask_fp
        
        #=======================================================================
        # extract clump data
        #=======================================================================
        with rio.open(clump_fp, mode='r') as ds: 
            profile = ds.profile
                       
            mar = load_array(ds, masked=True)
            assert_partial_wet(mar.mask, 'expects some nodata cells on the clump result')
            clump_ar = np.where(mar.mask, np.nan, mar.data)
            
            #identify the largest clump
            vals_ar, counts_ar = np.unique(clump_ar, return_counts=True, equal_nan=True)
            
            assert len(vals_ar)>1, f'wbt.clump failed to identify enough clusters\n    {clump_fp}'
            
            #===================================================================
            # max_clump_id = int(pd.Series(counts_ar, index=vals_ar).sort_values(ascending=False
            #             ).reset_index().dropna(subset='index').iloc[0, 0])
            #===================================================================
            
            clump_df = pd.Series(counts_ar, index=vals_ar).sort_values(ascending=False
                        ).rename('pixel_cnt').reset_index().dropna(subset='index')
                        
        #===================================================================
        # area-based selection
        #===================================================================
        log.debug(method)
        if method == 'area':
            
            clump_ids = clump_df.iloc[0:clump_cnt, 0].values
            

        
        #=======================================================================
        # pixel-based selection
        #=======================================================================
        elif method=='pixel':
            assert isinstance(wse_raw_fp, str), 'for method=pixel must pass wse_raw_fp'
            assert os.path.exists(wse_raw_fp)
            log.debug(f'filtering clumps w/ min_pixel_frac={min_pixel_frac}')
            
            #===================================================================
            # #drop some small clumps
            #===================================================================
            """because polygnoizing is very slow... want to drop any super small groups"""
            #select using 100 or some fraction
            bx = clump_df['pixel_cnt']>min(min_pixel_frac*clump_df['pixel_cnt'].sum(), 100)
            clump_ids = clump_df[bx].iloc[:,0].values
            
            log.info(f'pre-selected {bx.sum()}/{len(bx)} clumps for pixel selection')
            
            # build a mask of this
            bool_ar = np.isin(clump_ar, clump_ids)
            
            #write as mask
            clump_fp1 = write_array(np.where(bool_ar, clump_ar, np.nan), 
                                    ofp=os.path.join(tmp_dir, 'clump_mask_pre-filter.tif'), 
                                    masked=False, **profile)
            
            log.debug(f'wrote filtered clump mas to \n    {clump_fp1}')
            
            #===================================================================
            # polygonize clumps
            #===================================================================
            clump_vlay_fp = os.path.join(tmp_dir, 'clump_raster_to_vector_polygons.shp') #needs to be a shapefile
            assert self.raster_to_vector_polygons(clump_fp1, clump_vlay_fp) == 0
            log.debug(f'vectorized clumps to \n    {clump_vlay_fp}')
 
            #===================================================================
            # intersect of clumps and wse coarse
            #===================================================================
            wse_raw_pts = os.path.join(tmp_dir, 'wse_raw_points.shp') #needs to be a shapefile
            assert self.raster_to_vector_points(wse_raw_fp, wse_raw_pts) == 0
            
            #load both vector layers
            clump_poly_gdf = gpd.read_file(clump_vlay_fp)
            wse_pts_gdf = gpd.read_file(wse_raw_pts)
 
            
            clump_ids = clump_poly_gdf.sjoin(wse_pts_gdf, how='inner', predicate='intersects')['VALUE_left'].unique()
            
 
            
            log.info(f'selected {bx.sum()}/{len(clump_df)} clumps by pixel intersect')
 
            
 
            
            
        else:
            raise KeyError(method)
        
        # build a mask of this
        clump_bool_ar = np.isin(clump_ar, clump_ids)
        
        assert_partial_wet(clump_bool_ar)
        log.info(f'found clumps of {len(vals_ar)} with {clump_bool_ar.sum()}/{clump_bool_ar.size} unmasked cells' + \
                 '(%.2f)' % (clump_bool_ar.sum() / clump_bool_ar.size))
        
        meta_d.update({'clump_cnt':len(counts_ar), 'clump_max_size':clump_bool_ar.sum()})
            
        #=======================================================================
        # construct filter mask from clump selection
        #=======================================================================
        with rio.open(wse_fp, mode='r') as ds:
            profile = ds.profile
            wse_ar = load_array(ds, masked=True)
            
            #rebuild with union on mask
            wse_ar1 = ma.array(wse_ar.data, mask = np.logical_or(
                np.invert(clump_bool_ar), #not in the clump
                wse_ar.mask, #dry
                ), fill_value=wse_ar.fill_value)
            
 
        #=======================================================================
        # #write
        #=======================================================================
        assert_wse_ar(wse_ar1)
        write_array(wse_ar1, ofp=ofp, masked=False, **profile)
        
        meta_d.update({'raw_mask':wse_ar.mask.sum(), 'clump_filtered_mask':wse_ar1.mask.sum()})
        assert meta_d['raw_mask']<=meta_d['clump_filtered_mask']
 
        #=======================================================================
        # wrap
        #=======================================================================
        tdelta = (now()-start).total_seconds()
        meta_d['tdelta'] = tdelta
        meta_d['ofp'] = ofp
        log.info(f'wrote {wse_ar1.shape} in {tdelta:.2f} secs to \n    {ofp}')
        
        return ofp, meta_d
    

