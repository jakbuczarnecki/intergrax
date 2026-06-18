#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""RAG-MAINT-02 — export concurrent retrieval load/soak SLO report for nightly CI."""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
GOLDEN_CASES = REPO_ROOT / "tests" / "fixtures" / "rag_golden" / "retrieval_cases.json"
DEFAULT_OUTPUT = REPO_ROOT / "build" / "rag" / "load_soak_report.json"


def main() -> int:
    from intergrax.rag.evaluation.golden_harness import load_golden_cases
    from intergrax.rag.evaluation.load_soak import (
        build_soak_retrieval_service,
        export_load_soak_report,
        run_retrieval_load_soak,
        soak_queries_from_golden_cases,
    )

    if not GOLDEN_CASES.is_file():
        print(f"rag load soak report: missing golden cases at {GOLDEN_CASES}", file=sys.stderr)
        return 1

    cases = load_golden_cases(GOLDEN_CASES)
    queries = soak_queries_from_golden_cases(cases)
    if not queries:
        print("rag load soak report: no retrieval scenarios in golden cases", file=sys.stderr)
        return 1

    service = build_soak_retrieval_service(cases)
    result = run_retrieval_load_soak(service, queries)

    out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUTPUT
    export_load_soak_report(result, out_path, generated_at=datetime.now(UTC).isoformat())

    print(f"rag load soak report: passed={result.passed} p95_ms={result.p95_latency_ms:.2f}")
    print(f"  artifact: {out_path}")
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
