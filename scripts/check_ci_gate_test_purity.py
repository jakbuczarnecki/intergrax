#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""Fail when gate-marked unit tests use infra-heavy patterns without ``no_ci`` (CI purity gate)."""

from __future__ import annotations

import re
import sys
from pathlib import Path

SCAN_ROOT = Path("tests/unit")

FORBIDDEN: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("fastapi_testclient", re.compile(r"fastapi\.testclient|\bTestClient\b")),
    ("fastapi_app", re.compile(r"from fastapi import FastAPI|\bFastAPI\s*\(")),
    ("sleep", re.compile(r"\btime\.sleep\(|\basyncio\.sleep\(")),
    ("threading", re.compile(r"\bthreading\.(Lock|Thread|Event|Semaphore)\(")),
    ("asyncio_sync", re.compile(r"\basyncio\.(Lock|Semaphore|Event)\(")),
    (
        "sqlite",
        re.compile(
            r"sqlite3\.connect|Sqlite[A-Za-z]*Store|SQLite[A-Za-z]*Store"
            r"|:memory:|checkpoints\.db|trace\.db|events\.db|aiosqlite"
        ),
    ),
    ("http_client", re.compile(r"\bhttpx\.(get|post|Client)\(|\brequests\.(get|post)\(")),
    ("subprocess_server", re.compile(r"subprocess\.(Popen|run).*uvicorn")),
    ("mcp_runtime", re.compile(r"\bFastMCP\s*\(|couple_fastapi_with_mcp\s*\(|build_[a-z_]+_mcp_server\s*\(")),
)


def _has_gate_marker(text: str) -> bool:
    return "pytest.mark.gate" in text or "@pytest.mark.gate" in text


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    root = repo_root / SCAN_ROOT
    violations: list[tuple[str, list[str]]] = []
    for path in sorted(root.rglob("test_*.py")):
        text = path.read_text(encoding="utf-8")
        if not _has_gate_marker(text):
            continue
        if "pytest.mark.no_ci" in text:
            continue
        hits = sorted({name for name, pat in FORBIDDEN if pat.search(text)})
        if hits:
            rel = path.relative_to(repo_root).as_posix()
            violations.append((rel, hits))
    if violations:
        print("ci gate purity violations (gate tests with infra patterns, missing no_ci):")
        for rel, hits in violations:
            print(f"  {rel}: {', '.join(hits)}")
        return 1
    print("ci gate purity audit: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
