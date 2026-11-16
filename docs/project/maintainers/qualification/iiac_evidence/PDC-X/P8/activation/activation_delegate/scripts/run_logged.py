"""Run one command and write a log with command, HEAD, UTC start/end and exit code (activation delegate).

Usage (from the checkout root): python run_logged.py <log-name> -- <command...>
Writes <this folder>/../logs/<log-name>.log and echoes the output.
"""
import datetime
import subprocess
import sys
from pathlib import Path

LOGS = Path(__file__).resolve().parent.parent / "logs"


def utc() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def main() -> int:
    name = sys.argv[1]
    cmd = sys.argv[sys.argv.index("--") + 1 :]
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    LOGS.mkdir(parents=True, exist_ok=True)
    start = utc()
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    end = utc()
    body = (
        f"# command: {' '.join(cmd)}\n# HEAD: {head}\n# UTC start: {start}\n# UTC end: {end}\n"
        f"# python: {sys.executable}\n# ---- stdout ----\n{proc.stdout}"
        f"# ---- stderr ----\n{proc.stderr}# exit code: {proc.returncode}\n"
    )
    (LOGS / f"{name}.log").write_text(body, encoding="utf-8", newline="\n")
    sys.stdout.write(body)
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())
