#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Verify vendor SDK imports stay inside approved boundary modules (integrations, rag, agents)."""

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
        "weaviate",
        "pymilvus",
        "sentry_sdk",
    }
)

INTEGRATION_ALLOWED_SUFFIXES = (
    "/opens.py",
    "/rag_store.py",
    "/web_client.py",
    "/client.py",
    "/_shared/p3/factories.py",
    "/_shared/p3/clients.py",
)

RAG_ALLOWED_SUFFIXES = (
    "/opens.py",
    "/rag_store.py",
    "/client.py",
    "/parser_trace_exporter.py",
)


def _is_allowed(path: Path, *, scope: str) -> bool:
    posix = path.as_posix()
    if scope == "integrations":
        if "/integrations/providers/" not in posix:
            return True
        return any(posix.endswith(suffix) for suffix in INTEGRATION_ALLOWED_SUFFIXES)
    if scope == "rag":
        if "/rag/" not in posix:
            return True
        return any(posix.endswith(suffix) for suffix in RAG_ALLOWED_SUFFIXES)
    if scope == "agents":
        return "/agents/" not in posix
    return True


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


def _scan_scope(scope: str) -> list[str]:
    violations: list[str] = []
    for path in INTERGRAX.rglob("*.py"):
        if _is_allowed(path, scope=scope):
            continue
        for root, mod in _collect_imports(path):
            rel = path.relative_to(ROOT)
            violations.append(f"[{scope}] {rel}: import {mod} (vendor {root})")
    return violations


def main() -> int:
    violations: list[str] = []
    for scope in ("integrations", "rag", "agents"):
        violations.extend(_scan_scope(scope))
    if violations:
        print("Vendor import boundary violations:")
        for line in sorted(violations):
            print(f"  - {line}")
        return 1
    print("OK: no vendor imports outside approved boundary modules.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
