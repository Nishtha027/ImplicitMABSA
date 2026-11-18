@echo off
cd /d "%~dp0"
echo Starting the ImplicitMABSA model tester... (first start can take a minute)
echo When it says "Running on local URL", open http://127.0.0.1:7860 in your browser.
python test_model.py --models-dir models
pause
