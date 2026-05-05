@echo off
echo ============================================
echo   Irrigation Intelligente - Demarrage...
echo ============================================
cd /d "%~dp0"

call ..\.venv\Scripts\activate 2>nul
if errorlevel 1 (
    call .venv\Scripts\activate 2>nul
)

set MLFLOW_TRACKING_URI=sqlite:///mlflow.db
set MLFLOW_MODEL_URI=runs:/05dbc64a0c2e4236bf2e2f8c82f61f04/model

echo Interface disponible sur : http://127.0.0.1:5000/ui
echo Appuyez sur CTRL+C pour arreter.
echo.
python api\app.py
pause
