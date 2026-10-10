"""Run the repo's checks and report each: tests, lint, format, and the strict docs build."""

import argparse
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# Lines of a failed step's output to show.
TAIL_LINES = 25


def _steps(fast, site_dir):
    """Return `[(name, command)]` for the full check, or the inner loop when `fast`."""
    python = [sys.executable, "-m"]
    if fast:
        tests = [*python, "pytest", "tests/unit", "-m", "not slow", "-q"]
    else:
        tests = [*python, "pytest", "-n", "auto", "-q"]

    steps = [
        ("tests", tests),
        ("lint", [*python, "ruff", "check", "."]),
        ("format", [*python, "ruff", "format", "--check", "."]),
    ]
    if not fast:
        # Built into a temporary directory, so the repo's own site/ is left alone.
        steps.append(("docs", [*python, "mkdocs", "build", "--strict", "-d", site_dir]))
    return steps


def _run(name, command):
    """Run one step, print its result line, and return whether it passed."""
    start = time.monotonic()
    result = subprocess.run(command, cwd=REPO, capture_output=True, text=True, errors="replace")
    passed = result.returncode == 0
    print(f"{'pass' if passed else 'FAIL'}  {name:<7} {time.monotonic() - start:6.1f}s")
    if not passed:
        output = (result.stdout + result.stderr).strip().splitlines()
        for line in output[-TAIL_LINES:]:
            print(f"      {line}")
    return passed


def main():
    """Run every step, whatever the earlier ones did, and exit non-zero if any failed."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fast", action="store_true", help="Unit tests without the slow ones, and no docs build."
    )
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as site_dir:
        results = [_run(name, command) for name, command in _steps(args.fast, site_dir)]
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
