@echo off
chcp 65001 > nul
python -m pip install -r requirements.txt
python download_models.py
pause
