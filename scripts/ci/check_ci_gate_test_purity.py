#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""Fail when CI-selected unit tests (gate and not no_ci) use infra-heavy patterns."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

CI_MARKER = os.environ.get("INTERGRAX_CI_TEST_MARKER", "ci_smoke")

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
    ("subprocess_script", re.compile(r"subprocess\.(run|check_output|Popen)\(")),
    ("subprocess_server", re.compile(r"subprocess\.(Popen|run).*uvicorn")),
    ("mcp_runtime", re.compile(r"\bFastMCP\s*\(|couple_fastapi_with_mcp\s*\(|build_[a-z_]+_mcp_server\s*\(")),
)

PER_TEST_FORBIDDEN: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "tier3_host_bootstrap",
        re.compile(r"\bwire_application_environment\s*\("),
    ),
    (
        "lab_tool_bootstrap",
        re.compile(r"\bwire_lab_tools\s*\("),
    ),
    (
        "integration_catalog_bootstrap",
        re.compile(r"\bregister_default_integrations\s*\("),
    ),
    (
        "rag_stack_bootstrap",
        re.compile(r"\bcreate_default_rag_stack\s*\("),
    ),
    (
        "fleet_readiness_scan",
        re.compile(r"\bbuild_roster_readiness_report\s*\("),
    ),
    (
        "skill_registry_bootstrap",
        re.compile(r"\bbuild_application_skill_wiring\s*\("),
    ),
)

_FUNC_DEF = re.compile(r"^def (test_[a-zA-Z0-9_]+)\(", re.MULTILINE)


def _pytest_collect_roots(repo_root: Path) -> list[str]:
    if CI_MARKER == "ci_smoke":
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        from tests.unit.conftest import CI_SMOKE_DIR_PREFIXES, CI_SMOKE_FILES

        return [*CI_SMOKE_DIR_PREFIXES, *sorted(CI_SMOKE_FILES)]
    return ["tests/unit"]


def _collect_ci_tests(repo_root: Path) -> list[tuple[Path, str]]:
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            *_pytest_collect_roots(repo_root),
            "-m",
            CI_MARKER,
            "--collect-only",
            "-q",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode not in (0, 5):
        print(proc.stdout)
        print(proc.stderr, file=sys.stderr)
        raise SystemExit(f"pytest collect failed: {proc.returncode}")
    tests: list[tuple[Path, str]] = []
    seen: set[tuple[str, str]] = set()
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line.startswith("tests/") or "::" not in line:
            continue
        rel, node = line.split("::", 1)
        func = node.split("[", 1)[0]
        key = (rel, func)
        if key in seen:
            continue
        seen.add(key)
        tests.append((repo_root / rel, func))
    return tests


def _function_block(text: str, func_name: str) -> str:
    match = re.search(rf"^def {re.escape(func_name)}\(", text, re.MULTILINE)
    if match is None:
        return ""
    start = match.start()
    next_def = re.search(r"^def test_", text[match.end() :], re.MULTILINE)
    end = match.end() + next_def.start() if next_def else len(text)
    return text[start:end]


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    tests = _collect_ci_tests(repo_root)
    files = sorted({path for path, _ in tests})
    violations: list[tuple[str, list[str]]] = []
    per_test_violations: list[tuple[str, list[str]]] = []

    for path in files:
        text = path.read_text(encoding="utf-8")
        hits = sorted({name for name, pat in FORBIDDEN if pat.search(text)})
        if hits:
            rel = path.relative_to(repo_root).as_posix()
            violations.append((rel, hits))

    for path, func_name in tests:
        block = _function_block(path.read_text(encoding="utf-8"), func_name)
        if not block:
            continue
        hits = sorted({name for name, pat in PER_TEST_FORBIDDEN if pat.search(block)})
        if hits:
            rel = path.relative_to(repo_root).as_posix()
            per_test_violations.append((f"{rel}::{func_name}", hits))

    if violations or per_test_violations:
        print(f"ci gate purity violations ({CI_MARKER!r}, {len(files)} files, {len(tests)} tests):")
        for rel, hits in violations:
            print(f"  {rel}: {', '.join(hits)}")
        for nodeid, hits in per_test_violations:
            print(f"  {nodeid}: {', '.join(hits)}")
        return 1
    print(f"ci gate purity audit: OK ({len(files)} CI-selected test modules, {len(tests)} tests)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
