"""Central configuration. Paths can be overridden with environment variables."""
import os
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
PKG_DIR = SRC_DIR.parent
# student_resource/ root (holds dataset/ and output/)
ROOT = Path(os.environ.get("ER_ROOT", PKG_DIR.parent.parent))
DATA_DIR = Path(os.environ.get("ER_DATA", ROOT / "dataset"))
OUTPUT_DIR = Path(os.environ.get("ER_OUTPUT", ROOT / "output"))
CACHE_DIR = Path(os.environ.get("ER_CACHE", Path.home() / "er_work" / "cache"))

SEED = 42
VAL_FRAC = 0.10
N_THREADS = os.cpu_count() or 4

CACHE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
