from pathlib import Path

# Compute paths relative to installed source tree for reproducibility.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA = PROJECT_ROOT / "data"
ARTIFACTS = PROJECT_ROOT / "artifacts"

def ensure_dirs():
    for path in [DATA / "demo", DATA / "real", ARTIFACTS]:
        path.mkdir(parents=True, exist_ok=True)
