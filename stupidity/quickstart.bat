@echo off
REM Quick Start - Semantic Mutation Engine (Windows)

echo.
echo ╔════════════════════════════════════════════════════════════════╗
echo ║  AI Spectra: Semantic Mutation Engine                          ║
echo ║  Fast Start Guide (Windows)                                    ║
echo ╚════════════════════════════════════════════════════════════════╝
echo.

REM Check if venv exists
if not exist "venv" (
    echo 📦 Creating Python virtual environment...
    python -m venv venv
    
    echo 📥 Activating venv...
    call venv\Scripts\activate.bat
    
    echo 📚 Installing dependencies...
    pip install -r requirements.txt
) else (
    echo ✓ Virtual environment found
    call venv\Scripts\activate.bat
)

echo.
echo 🚀 Choose what to run:
echo.
echo 1) Discover Connections (Indian History + Aerospace)
echo    python explore_facts.py --visualize
echo.
echo 2) Extract and Analyze Web Content
echo    python fetch_and_walk.py --urls ^<url1^> ^<url2^> --visualize
echo.
echo 3) Run Original Pipeline
echo    python main.py --iterations 5
echo.
echo 4) View Generated Hypotheses
echo    python -c "import json; data = json.load(open('data/processed/generated.json')); [print(f\"{r['generated_text']}\") for r in sorted(data, key=lambda x: x.get('score', 0), reverse=True)[:5]]"
echo.
echo 📖 Full documentation: see README.md
echo.
