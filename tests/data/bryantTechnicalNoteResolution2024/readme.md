# Ahr test data

from:

Bryant, S., Schumann, G., Apel, H., Kreibich, H., and Merz, B.: Technical Note: Resolution enhancement of flood inundation grids, Hydrology and Earth System Sciences, 28, 575–588, https://doi.org/10.5194/hess-28-575-2024, 2024.


## wse_04m
water surface elevations

copied from `l:\02_WORK\NRC\2207_dscale\99_Arch\up\data\fdsc\r04_b4_i05_0508\wd_max_WSE.tif`

bbox (xmin, ymin, xmax, ymax):
     32357390.000000000, 5596180.000000000, 32370830.000000000, 5602836.000000000 [EPSG:4647]
    3360 w x 1664 h = 5591040 (4.00000000, 4.00000000)
    IMAGE_STRUCTURE: {'COMPRESSION': 'LZW', 'INTERLEAVE': 'BAND'}, NoDataValue: -9999.0
    BLOCKSIZE:[3360, 1]
  
## dem_04m.tif

copied from 'l:\02_WORK\NRC\2207_dscale\99_Arch\up\data\fdsc\r04_b4_i05_0508\dem005_r04_aoi13_0415.asc'
    
gdal_translate \
  "tests/data/bryantTechnicalNoteResolution2024/dem005_r04_aoi13_0415.tif" \
  "tests/data/bryantTechnicalNoteResolution2024/dem005_r04_aoi13_0415_f32_lerc.tif" \
  -ot Float32 \
  -co COMPRESS=LERC_DEFLATE \
  -co PREDICTOR=2 \
  -co ZLEVEL=6 \
  -co MAX_Z_ERROR=0.001


## wse_32m.tif
copied from `l:\02_WORK\NRC\2207_dscale\99_Arch\up\data\fdsc\r32_b10_i65_0511\wd_max_WSE.tif`

## dem_32m

gdal_translate \
  "tests/data/bryantTechnicalNoteResolution2024/dem005_r32_aoi13_0415.asc" \
  "tests/data/bryantTechnicalNoteResolution2024/dem_32m.tif" \
  -ot Float32 \
  -co COMPRESS=LERC_DEFLATE \
  -co PREDICTOR=2 \
  -co ZLEVEL=6 \
  -co MAX_Z_ERROR=0.001

/home/cefect/LS/09_REPOS/04_TOOLS/FloodDownscaler2/tests/data/bryantTechnicalNoteResolution2024/dem005_r32_aoi13_0415.asc