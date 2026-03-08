@echo off
REM Batch file to run local webcam demo (no API needed)
REM For demo when network is unstable

echo ================================================
echo   WEBCAM DEMO - LOCAL (No API/Network)
echo ================================================
echo.
echo Starting demo...
echo Press Q to quit, R to record, S for screenshot
echo.

cd /d "%~dp0"
python demo_webcam_local.py

if errorlevel 1 (
    echo.
    echo ================================================
    echo ERROR: Demo failed to run
    echo ================================================
    pause
) else (
    echo.
    echo Demo ended successfully.
)

pause
