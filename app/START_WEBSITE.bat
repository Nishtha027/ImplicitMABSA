@echo off
cd /d "%~dp0"
set MODELS=models
if not exist "%MODELS%" set MODELS=..\models
echo Starting the ImplicitMABSA website... (first start can take a minute while the models load)
echo When it says "ImplicitMABSA is running", open http://127.0.0.1:8000 in your browser.
python server.py --models-dir "%MODELS%"
pause
