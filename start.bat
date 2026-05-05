@echo off
echo ============================================
echo   Irrigation Intelligente - Demarrage...
echo ============================================
cd /d "%~dp0"

python --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo ERREUR : Python n'est pas installe.
    echo Installez Python 3.11 depuis https://www.python.org/downloads/
    echo Cochez "Add Python to PATH" pendant l'installation.
    echo.
    pause
    exit /b 1
)

if not exist ".venv\Scripts\activate" (
    echo Creation de l'environnement virtuel...
    python -m venv .venv
)

call .venv\Scripts\activate

echo Installation des dependances (1-2 minutes)...
pip install -r requirements_api.txt --quiet

set MLFLOW_TRACKING_URI=sqlite:///mlflow.db
set MLFLOW_MODEL_URI=runs:/05dbc64a0c2e4236bf2e2f8c82f61f04/model

echo.
echo Demarrage...
echo Ouverture : http://127.0.0.1:5000/ui
echo Appuyez sur CTRL+C pour arreter.
echo.
start "" http://127.0.0.1:5000/ui
python api\app.py
pause
