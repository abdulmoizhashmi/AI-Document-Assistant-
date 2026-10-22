@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Creating virtual environment...
  py -m venv .venv
  call ".venv\Scripts\activate.bat"
  python -m pip install --upgrade pip
  python -m pip install -r requirements.txt
  echo installed > ".venv\.installed"
) else (
  call ".venv\Scripts\activate.bat"
  if not exist ".venv\.installed" (
    echo Checking dependencies...
    python -m pip install -r requirements.txt
    echo installed > ".venv\.installed"
  )
)
if not exist ".env" (
  copy ".env.example" ".env" >nul
  echo Created .env from .env.example. Add your API key(s) before running.
  notepad .env
  pause
)
echo Starting AI Document Assistant...
streamlit run app.py
endlocal
