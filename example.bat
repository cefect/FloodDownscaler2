:: example windows batch script for using FloodDownscaler2's CLI
:: USE
:: 1) install/build FloodDownscaler2 per the readme
:: 2) configure the 'activate environment' section to properly activate your environment before the mian execution call (alternatively, you could call this batch script from your activated environment). 

:: inputs/outputs
SET WSE_FP=%~dp0\tests\data\ahr\wse2_clip.tif
SET DEM_FP=%~dp0\tests\data\ahr\dem1_clip.tif
SET OUT_DIR=%USERPROFILE%\FloodDownscaler2


:: activate the environment
call %~dp0\env\conda_activate.bat

:: execute the downscaler
python %SRC_DIR%\fdsc\main.py %DEM_FP% %WSE_FP% --out_dir %OUT_DIR%

:: launch the results window
start "" %OUT_DIR%

ECHO finished

::keep window open
cmd.exe /k
