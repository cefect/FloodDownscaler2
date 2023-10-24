

:: activate the environment
call %~dp0..\env\conda_activate.bat


coverage run -m pytest ./test_dsc.py

cmd.exe /k