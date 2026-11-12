# Run this in the project folder AFTER installing Python 3.11.
$ErrorActionPreference = 'Stop'
if (-not (Test-Path '.venv')) { py -3.11 -m venv .venv }
$py = '.\.venv\Scripts\python.exe'
& $py -m pip install -e '.[test]'
& $py -m swapscribe.cli all
& $py -m pytest -q
& $py -m streamlit run dashboard.py
