# FloodDownscaler

Tools for downscaling/disaggregating flood hazard grids

## TODO:
integrate coms
split case study from downscaling tool
split definitions from parameters (which should go in the case study)

## Use
see ./example

## Related projects
[FloodDownscaler](https://github.com/cefect/FloodDownscaler): original repo for HESS publication work

[FloodRescaler](https://github.com/cefect/FloodRescaler): public repo with QGIS processing script port (and aggregation tools)

[rimpy](https://git.gfz-potsdam.de/bryant/rimpy): Tools for building, calibrating, and visualizing RIM2D models
 
[2207_dscale2](https://github.com/cefect/2207_dscale2): **OLD** project for generating analog inundation grids with LISFLOOD. 

[FloodPolisher](https://github.com/cefect/FloodPolisher): mid-2022 inundation downscaling work using simple growth. pyqgis. Should incorporate a version of this into this project. 

[2112_agg_pub](https://github.com/cefect/2112_agg_pub): public repo of analysis for aggregation paper. 

## Submodules

PYTHONPATH:
PROJECT_DIR_NAME
PROJECT_DIR_NAME\whitebox-tools
PROJECT_DIR_NAME\whitebox-tools\target\release (need to build first)

 

whitebox tools (cefect's fork)
`git submodule https://github.com/cefect/whitebox-tools.git`

    v2.2.0
        git switch c8d03fc3154a34d2d2904491ee36a7ab8239289c --detached
        
    need to build this using rust (see below)
    



## Installation

build a python environment per ./environment.yml

build whitebox-tools using rust
    cargo build --release
    takes a while
    
add submodules to pythonpath (see above)

create and customize a ./definitions.py file (see below)


### definitions.py

```
import os

src_dir = os.path.dirname(os.path.abspath(__file__))
src_name = os.path.basename(src_dir)

#location of logging configuration file
logcfg_file=os.path.join(src_dir, r'coms\logger.conf')

#default working directory
wrk_dir = r'L:\10_IO\fdsc2'

#whitebox exe location
wbt_dir = os.path.join(src_dir, r'whitebox-tools\target\release')

#specify the latex install directory
os.environ['PATH'] += R";c:\Users\cefect\AppData\Local\Programs\MiKTeX\miktex\bin\x64"
```