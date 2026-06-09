#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""IDEAL-17.1 — reject long inline prompt strings in Tier-2 agent pipelines."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
AGENTS_DIR = REPO_ROOT / "agents"
MIN_INLINE_PROMPT_CHARS = 120


def _is_pipeline_file(path: Path) -> bool:
    return path.name == "pipeline.py" and "steps" in path.parts


def _long_string_nodes(source: str) -> list[tuple[int, int]]:
    tree = ast.parse(source)
    hits: list[tuple[int, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if len(node.value) >= MIN_INLINE_PROMPT_CHARS and "\n" in node.value:
                hits.append((node.lineno, len(node.value)))
    return hits


def main() -> int:
    violations: list[str] = []
    for path in sorted(AGENTS_DIR.rglob("pipeline.py")):
        if not _is_pipeline_file(path):
            continue
        rel = path.relative_to(REPO_ROOT)
        hits = _long_string_nodes(path.read_text(encoding="utf-8"))
        for lineno, length in hits:
            violations.append(f"{rel}:{lineno}: inline prompt-like string ({length} chars)")
    if violations:
        print("Inline prompt violations (use Prompt Registry):")
        for line in violations:
            print(f"  {line}")
        return 1
    print("OK: no inline prompt violations in agent pipelines")
    return 0


if __name__ == "__main__":
    sys.exit(main())
