"""Launcher for the bact-trait-cluster marimo dashboard."""

from __future__ import annotations

import argparse
import subprocess
import sys
from importlib import resources


def _dashboard_file() -> str:
    dash = resources.files("bacttraitcluster.dashboards").joinpath("app.py")
    with resources.as_file(dash) as p:
        return str(p)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bact-trait-cluster-dashboard")
    parser.add_argument("--edit", action="store_true")
    parser.add_argument("marimo_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)

    try:
        import marimo  # noqa: F401
    except Exception:
        print("marimo is not installed. Install: pip install 'bact-trait-cluster[gui]'", file=sys.stderr)
        return 2

    mode = "edit" if args.edit else "run"
    dash_path = _dashboard_file()
    cmd = [sys.executable, "-m", "marimo", mode, dash_path]
    extra = args.marimo_args
    if extra and extra[0] == "--":
        extra = extra[1:]
    cmd.extend(extra)
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
