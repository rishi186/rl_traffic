@echo off
set "PROFILE=%~1"
if not defined PROFILE set "PROFILE=smoke"
set "OUTPUT=%~2"
if not defined OUTPUT set "OUTPUT=results/campaign-%RANDOM%-%RANDOM%"
.venv\Scripts\python.exe scripts\run_experiments.py --config config.yaml --benchmark-profile "%PROFILE%" --output-dir "%OUTPUT%"
exit /b %ERRORLEVEL%
