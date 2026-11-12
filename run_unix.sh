#!/usr/bin/env bash
# Run this in the project folder AFTER installing a supported Python.
set -euo pipefail
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
python -m swapscribe.cli all
python -m pytest -q
python -m streamlit run dashboard.py
