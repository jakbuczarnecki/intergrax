#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""Regression gate: RAG OTel span registry and hot-path wiring (M-RAG.27)."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]

_SPAN_WIRING: dict[str, str] = {
    "rag.retrieve": "intergrax/rag/retrieval/retrieval_service.py",
    "rag.retrieve.single_pass": "intergrax/rag/retrieval/retrieval_service.py",
    "rag.ingest": "intergrax/rag/ingest/ingest_pipeline.py",
    "rag.ingest.load": "intergrax/rag/ingest/ingest_pipeline.py",
    "rag.ingest.chunk": "intergrax/rag/ingest/ingest_pipeline.py",
    "rag.ingest.index": "intergrax/rag/ingest/ingest_pipeline.py",
    "rag.ingest.graph_index": "intergrax/rag/ingest/ingest_pipeline.py",
}


def _verify_registry_matches_wiring() -> list[str]:
    from intergrax.rag.tracking.rag_spans import RAG_OTEL_SPAN_NAMES

    errors: list[str] = []
    registered = set(RAG_OTEL_SPAN_NAMES)
    wired = set(_SPAN_WIRING)
    if registered != wired:
        missing = sorted(wired - registered)
        extra = sorted(registered - wired)
        if missing:
            errors.append(f"registry missing span names: {missing}")
        if extra:
            errors.append(f"registry has unwired span names: {extra}")

    for span_name, rel_path in _SPAN_WIRING.items():
        source = (_REPO_ROOT / rel_path).read_text(encoding="utf-8")
        pattern = re.compile(rf'rag_span\(\s*["\']{re.escape(span_name)}["\']')
        if not pattern.search(source):
            errors.append(f"{rel_path} missing rag_span wiring for {span_name!r}")
    return errors


def main() -> int:
    errors = _verify_registry_matches_wiring()
    if errors:
        for err in errors:
            print(f"rag otel span registry audit: FAIL — {err}", file=sys.stderr)
        return 1

    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "tests/unit/rag/tracking/test_rag_otel_spans.py",
        "-q",
    ]
    result = subprocess.run(cmd, cwd=_REPO_ROOT, check=False)
    if result.returncode != 0:
        return result.returncode

    print("rag otel span registry audit: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
