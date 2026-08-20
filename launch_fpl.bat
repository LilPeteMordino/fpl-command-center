@echo off
title FPL Analytics Hub
echo ==============================================
echo    Starting FPL Analytics Hub & Solver...
echo ==============================================
echo.

:: Navigate to your project folder
cd /d "C:\Users\danth\OneDrive\Documents\CLAUDE\FPL"

:: Launch Streamlit via Python
python -m streamlit run app.py

pause