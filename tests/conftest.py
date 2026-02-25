import sys
from pathlib import Path

# Ensure editable-style imports without requiring package installation.
SRC = Path(__file__).resolve().parents[1] / "src"
if SRC.exists():
    sys.path.insert(0, str(SRC))
