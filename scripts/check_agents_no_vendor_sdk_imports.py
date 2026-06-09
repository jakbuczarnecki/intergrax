#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""IDEAL-2.2 / IDEAL-29.3 — Tier-2 agents must not import vendor ML/SDK packages."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
AGENTS_DIR = REPO_ROOT / "agents"

FORBIDDEN_ROOTS = frozenset(
    {
        "openai",
        "anthropic",
        "boto3",
        "torch",
        "tensorflow",
        "onnxruntime",
        "ultralytics",
        "cv2",
        "sklearn",
    }
)


def _forbidden_imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    hits: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", 1)[0]
                if root in FORBIDDEN_ROOTS:
                    hits.append(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            root = node.module.split(".", 1)[0]
            if root in FORBIDDEN_ROOTS:
                hits.append(node.module)
    return hits


def main() -> int:
    violations: list[str] = []
    for path in sorted(AGENTS_DIR.rglob("*.py")):
        rel = path.relative_to(REPO_ROOT)
        for mod in _forbidden_imports(path):
            violations.append(f"{rel}: imports forbidden vendor module {mod!r}")
    if violations:
        print("Vendor SDK import violations in agents/:")
        for line in violations:
            print(f"  {line}")
        return 1
    print("OK: no forbidden vendor SDK imports in agents/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
