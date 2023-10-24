:: example windows batch script for using FloodDownscaler2's CLI

:: inputs/outputs
SET WSE_FP=%~dp0\tests\data\ahr\wse2_clip.tif
SET DEM_FP=%~dp0\tests\data\ahr\dem1_clip.tif
SET OUT_DIR=%USERPROFILE%\FloodDownscaler2


:: activate the environment
call %~dp0\env\conda_activate.bat

:: execute the downscaler
python %~dp0\fdsc\main.py %DEM_FP% %WSE_FP% --out_dir %OUT_DIR%

:: launch the results window
start "" %OUT_DIR%

ECHO finished

::keep window open
cmd.exe /k
