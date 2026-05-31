#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Verify vendor SDK imports stay inside Integration Library boundary modules."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INTERGRAX = ROOT / "intergrax"

VENDOR_MODULES = frozenset(
    {
        "chromadb",
        "cohere",
        "openpyxl",
        "whisper",
        "yt_dlp",
        "pinecone",
        "qdrant_client",
        "redis",
        "boto3",
        "googleapiclient",
    }
)

ALLOWED_SUFFIXES = (
    "/opens.py",
    "/rag_store.py",
    "/web_client.py",
    "/client.py",
    "/_shared/p3/factories.py",
    "/_shared/p3/clients.py",
)


def _is_allowed(path: Path) -> bool:
    posix = path.as_posix()
    if "/integrations/providers/" not in posix:
        return True
    return any(posix.endswith(suffix) for suffix in ALLOWED_SUFFIXES)


def _collect_imports(path: Path) -> list[tuple[str, str]]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    hits: list[tuple[str, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in VENDOR_MODULES:
                    hits.append((root, alias.name))
        elif isinstance(node, ast.ImportFrom) and node.module:
            root = node.module.split(".")[0]
            if root in VENDOR_MODULES:
                hits.append((root, node.module))
    return hits


def main() -> int:
    violations: list[str] = []
    for path in INTERGRAX.rglob("*.py"):
        if _is_allowed(path):
            continue
        for root, mod in _collect_imports(path):
            rel = path.relative_to(ROOT)
            violations.append(f"{rel}: import {mod} (vendor {root})")
    if violations:
        print("Integration vendor import boundary violations:")
        for line in sorted(violations):
            print(f"  - {line}")
        return 1
    print("OK: no vendor imports outside integration boundary modules.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
