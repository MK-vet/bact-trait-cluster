from __future__ import annotations
import os
import sys
import subprocess
from pathlib import Path

def main() -> None:
    try:
        import marimo  # noqa: F401
    except Exception as e:
        print("marimo is not installed. Install with: pip install -e '.[gui]'", file=sys.stderr)
        raise SystemExit(2) from e

    app_path = Path(__file__).with_name("marimo_app.py")
    port = os.environ.get("SSUIS_GUI_PORT", "")
    cmd = [sys.executable, "-m", "marimo", "run", str(app_path)]
    if port:
        cmd += ["--port", str(port)]
    if "--" in sys.argv:
        idx = sys.argv.index("--")
        cmd += sys.argv[idx+1:]
    subprocess.run(cmd, check=False)

if __name__ == "__main__":
    main()
