# FloodDownscaler

Tools for downscaling/disaggregating flood hazard grids

![alt text](./icon/logo_flood_rescale_20230205_600dpi.png)

 

## Use
To downscale a coarse WSE using a fine resolution DEM, the CLI in main.py can be used:
```
python fdsc\main.py <path_to_DEM> <path_to_WSE>
```
an example application with this CLI is provided in [example.bat](./example.bat)

Alternatively, the `downscale` function can be called within a python script:
```
from fdsc.main import downscale
downscale(<path_to_DEM>, <path_to_WSE>)
>>> <path_to_WSE1>, {metadata}
```

## Related projects
[FloodDownscaler](https://github.com/cefect/FloodDownscaler): original repo for HESS publication work. decided to make this new repo as something shareable and more streamlined. 

[FloodRescaler](https://github.com/cefect/FloodRescaler): public repo with QGIS processing script port (and aggregation tools)

[rimpy](https://git.gfz-potsdam.de/bryant/rimpy): Tools for building, calibrating, and visualizing RIM2D models
 
[2207_dscale2](https://github.com/cefect/2207_dscale2): **OLD** project for generating analog inundation grids with LISFLOOD. 

[FloodPolisher](https://github.com/cefect/FloodPolisher): mid-2022 inundation downscaling work using simple growth. pyqgis. Should incorporate a version of this into this project. 

[2112_agg_pub](https://github.com/cefect/2112_agg_pub): public repo of analysis for aggregation paper. 


    



## Installation

build a python environment per ./environment.yml

create and customize a ./definitions.py file (see below)

### clone and build whitebox-tools
project is setup to use whitebox-tools v2.2.0 as a submodule
this can be achieved a number of ways, but the cleanest (also hardest) is to:
- clone whitebox-tools into the repo as a submodule and point to the v2.2.0 release tag.
`git submodule add -b v2.2.0 https://github.com/cefect/whitebox-tools.git`
- compile the tools. call the below within the newly cloned submodule folder. this may take a while and requires you to have rust installed (see whitebox-tools documentation for more info)
`cargo build --release`
- update your python path to the below




### PYTHONPATH
replace PROJECT_DIR_NAME with the path to your repo. The last directory is created by building whitebox-tools.
```
PROJECT_DIR_NAME
PROJECT_DIR_NAME\whitebox-tools
PROJECT_DIR_NAME\whitebox-tools\target\release 
```
 


### definitions.py

```
import os

src_dir = os.path.dirname(os.path.abspath(__file__))
src_name = os.path.basename(src_dir)

#location of logging configuration file
logcfg_file=os.path.join(src_dir, r'hp\logger.conf')

#default working directory
wrk_dir = r'L:\10_IO\fdsc2'

#whitebox exe location
wbt_dir = os.path.join(src_dir, r'whitebox-tools\target\release')
 
```