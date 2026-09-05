#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""Stage 16 — canonical authoring surface conformance gate (bounded scope)."""

from __future__ import annotations

import argparse
import re
from collections.abc import Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

CANONICAL_SURFACES: tuple[Path, ...] = (
    REPO_ROOT / "docs" / "project" / "technical" / "guides" / "AGENT_CREATION_GUIDE.md",
    REPO_ROOT / "intergrax" / "scaffold" / "new_agent.py",
    REPO_ROOT / "intergrax" / "scaffold" / "doc_templates.py",
)

GUIDE_BOUNDED_START = "## 1. Mental model"
GUIDE_BOUNDED_END = "## Step 6 - Inspect traces"
GUIDE_EXTRA_MARKERS: tuple[tuple[str, str | None], ...] = (
    ("## Anti-patterns", "## Instructions for LLM coding agents"),
    ("## Instructions for LLM coding agents", None),
)

_FORBIDDEN_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("AgentRegistry()", re.compile(r"(?<![`])\bAgentRegistry\s*\(\s*\)(?!`)")),
    ("registry.register(", re.compile(r"(?<![`])\bregistry\.register\s*\(")),
    ("NexusLoop(", re.compile(r"(?<![`])\bNexusLoop\s*\(")),
    (
        "AgentRegistry.register() registration rule",
        re.compile(r"integrates through `AgentRegistry\.register\(\)`"),
    ),
    (
        "Run via NexusLoop quickstart",
        re.compile(r"Run via NexusLoop"),
    ),
)

_ALLOW_SUBSTRINGS: tuple[str, ...] = (
    "do not",
    "do **not**",
    "**not**",
    " not ",
    "never",
    "forbidden",
    "anti-pattern",
    "historical",
    "not a canonical",
    "not ad-hoc",
    "must not",
    "without the distribution",
)


def _surface_paths() -> tuple[Path, ...]:
    agent_readmes = tuple(sorted((REPO_ROOT / "agents").glob("*/README.md")))
    return CANONICAL_SURFACES + agent_readmes


def _line_allowed(line: str) -> bool:
    lowered = line.lower()
    return any(token in lowered for token in _ALLOW_SUBSTRINGS)


def _guide_bounded_chunks(text: str) -> list[str]:
    chunks: list[str] = []
    start = text.index(GUIDE_BOUNDED_START)
    end = text.index(GUIDE_BOUNDED_END, start)
    chunks.append(text[start:end])
    for begin_marker, end_marker in GUIDE_EXTRA_MARKERS:
        begin = text.index(begin_marker)
        if end_marker is None:
            chunks.append(text[begin:])
            continue
        stop = text.index(end_marker, begin + len(begin_marker))
        chunks.append(text[begin:stop])
    return chunks


def scan_surface(path: Path) -> list[str]:
    violations: list[str] = []
    text = path.read_text(encoding="utf-8")
    if path.name == "AGENT_CREATION_GUIDE.md":
        lines: list[tuple[int, str]] = []
        offset = 0
        for chunk in _guide_bounded_chunks(text):
            for line_no, line in enumerate(chunk.splitlines(), start=1):
                lines.append((line_no, line))
    else:
        lines = [(line_no, line) for line_no, line in enumerate(text.splitlines(), start=1)]

    for line_no, line in lines:
        if _line_allowed(line):
            continue
        for label, pattern in _FORBIDDEN_PATTERNS:
            if pattern.search(line):
                rel = path.relative_to(REPO_ROOT)
                violations.append(
                    f"CANONICAL_AUTHORING_VIOLATION: {rel}:{line_no}: {label}: {line.strip()}"
                )
    return violations


def audit_repository(repo_root: Path = REPO_ROOT) -> list[str]:
    violations: list[str] = []
    for path in _surface_paths():
        if not path.is_file():
            violations.append(f"CANONICAL_AUTHORING_MISSING_SURFACE: {path.relative_to(repo_root)}")
            continue
        violations.extend(scan_surface(path))
    return violations


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(list(argv) if argv is not None else [])
    violations = audit_repository()
    if violations:
        for item in violations:
            print(item.encode("ascii", errors="backslashreplace").decode("ascii"))
        return 1
    print("Canonical authoring surface conformance: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
