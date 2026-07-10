"""Run the cookbook method demo with local example data."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

METHOD_DIR = Path(__file__).resolve().parent.parent


def main() -> None:
    subprocess.run(
        [sys.executable, str(METHOD_DIR / "4-run-method.py"), "--query", "Where does vendor onboarding require security review?"],
        cwd=METHOD_DIR,
        check=True,
    )


if __name__ == "__main__":
    main()
