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
[FloodDownscaler](https://github.com/cefect/FloodDownscaler): original repo for HESS publication work

[FloodRescaler](https://github.com/cefect/FloodRescaler): public repo with QGIS processing script port (and aggregation tools)

[rimpy](https://git.gfz-potsdam.de/bryant/rimpy): Tools for building, calibrating, and visualizing RIM2D models
 
[2207_dscale2](https://github.com/cefect/2207_dscale2): (**DISCONTINUED**) project for generating analog inundation grids with LISFLOOD. 

[FloodPolisher](https://github.com/cefect/FloodPolisher): (**DISCONTINUED**) mid-2022 inundation downscaling work using simple growth. pyqgis. Should incorporate a version of this into this project. 

[2112_agg_pub](https://github.com/cefect/2112_agg_pub): public repo of analysis for aggregation paper. 

## Installation

build a python environment per ./environment.yml

create and customize a ./definitions.py file (see below)

set paths for whitebox-tools v2.2.0

### setting up whitebox-tools
Some of the functions depend on whitebox-tools (wbt) python API. For these to work, installing FloodDownscaler2 requires installing whitebox-tools (if you haven't already) and telling FloodDownscaler2 where to find it. Below is some guidance/instructions for configuring wbt; however, there are a few ways to do this and not all ways work on all systems. 

#### wbt is already installed
ammend the `wbt_dir` variable in `definitions.py` to point to the directory contining your  whitebox_tools.exe 
ammend your PYTHONPATH as shown below

#### install wbt from precompiled binary
compiled binaries are available [here](https://www.whiteboxgeo.com/download-direct/) for common systems. 

#### building wbt from source



#### building wbt from source as a submodule

This is the preferred method as it provides greater version control. Typically, FloodDownscaler2 project is setup to use whitebox-tools v2.2.0 as a submodule. 
Typical workflow:
- clone whitebox-tools into the repo as a submodule and point to the v2.2.0 release tag.
`git submodule add -b v2.2.0 https://github.com/cefect/whitebox-tools.git`
- compile the tools. call the below within the newly cloned submodule folder. this may take a while and requires you to have rust installed (see whitebox-tools documentation for more info)
`cargo build --release`
- update paths per the above




### PYTHONPATH
The project requires the PYTHONPATH to include the source directory, and two whitebox-tools directories. 

Below is an example for a project that builds whitebox-tools from scratch. Replace PROJECT_DIR_NAME with the path to your repo. The last directory is created by building whitebox-tools.
```
PROJECT_DIR_NAME
PROJECT_DIR_NAME\whitebox-tools
PROJECT_DIR_NAME\whitebox-tools\target\release 
```

### definitions.py

```

import os
#default working directory
wrk_dir = r'L:\10_IO\fdsc2'

#whitebox exe location
wbt_dir = os.path.join(src_dir, r'whitebox-tools\target\release')
 
```