"""Offline, safe environment diagnostic for YieldPulse AI."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parent
REQUIRED = ("numpy", "pandas", "scipy", "sklearn", "joblib", "requests", "plotly", "streamlit")


def mark(ok: bool) -> str:
    return "OK" if ok else "MISSING"


def main() -> int:
    print("SwapRisk-and-FedNews-AI — Local Diagnostic")
    print(f"OS: {platform.platform()}")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Python interpreter: {sys.executable}")
    print(f"Working directory: {Path.cwd()}")
    print(f"Project root: {ROOT}")
    version_ok = (3, 10) <= sys.version_info[:2] < (3, 14)
    print(f"Python supported (3.10–3.13): {mark(version_ok)}")
    print("\nDependencies:")
    dep_ok = True
    for package in REQUIRED:
        exists = importlib.util.find_spec(package) is not None
        dep_ok = dep_ok and exists
        print(f"  {mark(exists):7} {package}")
    installed = importlib.util.find_spec("swapscribe") is not None
    print(f"  {mark(installed):7} swapscribe editable project")
    print("\nProject files:")
    for filename in (
        "pyproject.toml", "dashboard.py", "data/demo/market_rates.csv",
        "data/demo/portfolio.csv", "data/demo/news.csv",
        "data/real/treasury_2025.csv", "data/real/treasury_2026.csv",
        "data/real/fed_news.csv", "data/real/sentiment_train.csv",
        "artifacts/sentiment_real.joblib", "artifacts/sentiment_real_eval.json"
    ):
        path = ROOT / filename
        print(f"  {mark(path.is_file()):7} {filename}")
    if not version_ok or not dep_ok or not installed:
        print("\nACTION: From project root, install dependencies using THIS interpreter:")
        print(f'  "{sys.executable}" -m pip install -e ".[test]"')
        return 1
    print("\nEnvironment OK. Next steps:")
    print(f'  "{sys.executable}" -m pytest -q')
    print(f'  "{sys.executable}" -m swapscribe.cli all')
    print(f'  "{sys.executable}" -m streamlit run dashboard.py')
    print("Note: missing REAL files are expected until you run the fetch commands.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
