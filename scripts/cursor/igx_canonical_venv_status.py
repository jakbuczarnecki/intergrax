#!/usr/bin/env python3
"""Report canonical .venv health and processes that may block uv sync on Windows."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_VENV_PY = _REPO / ".venv" / "Scripts" / "python.exe"


def _probe_canonical() -> dict[str, bool]:
    if not _VENV_PY.is_file():
        return {"venv_present": False, "tiktoken_get_encoding": False, "zstd_compressor": False}
    code = (
        "import tiktoken, zstandard; "
        "print(bool(getattr(tiktoken,'get_encoding',None))); "
        "print(bool(getattr(zstandard,'ZstdCompressor',None)))"
    )
    proc = subprocess.run(
        [_VENV_PY, "-c", code],
        cwd=_REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    lines = [line.strip() for line in proc.stdout.splitlines()]
    tik = lines[0] == "True" if lines else False
    zst = lines[1] == "True" if len(lines) > 1 else False
    return {
        "venv_present": True,
        "tiktoken_get_encoding": tik,
        "zstd_compressor": zst,
        "probe_exit": proc.returncode,
    }


def _intergrax_python_holders() -> list[dict[str, str | int]]:
    if sys.platform != "win32":
        return []
    ps = (
        "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
        "Where-Object { $_.ExecutablePath -like '*intergrax*' } | "
        "Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress"
    )
    proc = subprocess.run(
        ["powershell", "-NoProfile", "-Command", ps],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0 or not proc.stdout.strip():
        return []
    raw = json.loads(proc.stdout)
    if isinstance(raw, dict):
        raw = [raw]
    out: list[dict[str, str | int]] = []
    for row in raw:
        out.append(
            {
                "pid": int(row.get("ProcessId", 0)),
                "command_line": str(row.get("CommandLine") or "")[:500],
            }
        )
    return out


def main() -> int:
    report = {
        "repo": str(_REPO),
        "canonical_venv": _probe_canonical(),
        "blocking_python_processes": _intergrax_python_holders(),
    }
    healthy = report["canonical_venv"].get("tiktoken_get_encoding") and report["canonical_venv"].get("zstd_compressor")
    report["canonical_healthy"] = healthy
    print(json.dumps(report, indent=2))
    return 0 if healthy else 1


if __name__ == "__main__":
    raise SystemExit(main())
