#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""Measure typing/reflection metrics on the frozen contract surface (intergrax/contracts)."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SURFACE = REPO_ROOT / "intergrax" / "contracts"


def _count_type_name_in_tree(node: ast.expr | None, type_name: str) -> int:
    if node is None:
        return 0
    total = 0
    for child in ast.walk(node):
        if isinstance(child, ast.Name) and child.id == type_name:
            total += 1
        if isinstance(child, ast.Attribute) and child.attr == type_name:
            total += 1
    return total


def _file_fingerprint(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def measure_contract_file(path: Path) -> dict[str, int]:
    text = path.read_text(encoding="utf-8")
    source = ast.parse(text)
    counts = {"any": 0, "object": 0, "getattr": 0, "setattr": 0, "hasattr": 0}
    for node in ast.walk(source):
        annotations: list[ast.expr | None] = []
        if isinstance(node, ast.arg):
            annotations.append(node.annotation)
        elif isinstance(node, ast.AnnAssign):
            annotations.append(node.annotation)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            annotations.append(node.returns)
        for ann in annotations:
            counts["any"] += _count_type_name_in_tree(ann, "Any")
            counts["object"] += _count_type_name_in_tree(ann, "object")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            name = node.func.id
            if name in ("getattr", "setattr", "hasattr"):
                counts[name] += 1
    return counts


def measure_contract_surface(surface_root: Path = DEFAULT_SURFACE) -> dict[str, Any]:
    totals = {"any": 0, "object": 0, "getattr": 0, "setattr": 0, "hasattr": 0, "files_scanned": 0}
    fingerprints: dict[str, str] = {}
    for path in sorted(surface_root.rglob("*.py")):
        if not path.is_file():
            continue
        rel = path.relative_to(REPO_ROOT).as_posix()
        rel_counts = measure_contract_file(path)
        for key in ("any", "object", "getattr", "setattr", "hasattr"):
            totals[key] += rel_counts[key]
        totals["files_scanned"] += 1
        fingerprints[rel] = _file_fingerprint(path.read_text(encoding="utf-8"))
    return {"metrics": totals, "file_fingerprints": fingerprints}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Emit JSON on stdout")
    parser.add_argument("--surface", type=Path, default=DEFAULT_SURFACE)
    args = parser.parse_args()
    report = measure_contract_surface(args.surface)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        metrics = report["metrics"]
        print(
            "frozen contract typ:",
            f"files={metrics['files_scanned']}",
            f"any={metrics['any']}",
            f"object={metrics['object']}",
            f"getattr={metrics['getattr']}",
            f"setattr={metrics['setattr']}",
            f"hasattr={metrics['hasattr']}",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
