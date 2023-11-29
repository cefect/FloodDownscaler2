"""2023-11-29: running pluvial grid from Livio"""


dem_fp=r'l:\10_IO\FloodRescaler\issues\10\Data\DEM_channel_corrected_clipped_aoiT01.tif'
wse_fp=r'l:\10_IO\FloodRescaler\issues\10\Data\WSE_rp100_aoiT01.tif'

import os
from fdsc.control import Dsc_Session
from definitions import wrk_dir

 

with Dsc_Session(run_name='fdsc2', relative=True, 
                 out_dir=os.path.join(wrk_dir, 'outs','issue', 'i10')) as ses:

    ses.run_dsc(dem_fp, wse_fp, method='CostGrow', write_meta=False,
                rkwargs = dict(
                    clump_method='pixel', #select flooding clump by intersect
                    loss_frac=0.1, #add a 10percent decay
                    )
 
                )