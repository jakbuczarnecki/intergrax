#!/usr/bin/env python3
"""EC-W3-12 reconciliation emitter — recompute dispositions (does not run validator)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from _evidence_common import compute_canonical_evidence_projection, resolve_execution_context

RUN_ROOT = Path(__file__).resolve().parent


def _die(msg: str) -> None:
    print(f"RECONCILE_FAIL: {msg}", file=sys.stderr)
    raise SystemExit(1)


def emit(ctx) -> None:
    baseline_path = ctx.run_root / "BASELINE_BINDING.json"
    if not baseline_path.is_file():
        _die("missing BASELINE_BINDING.json for qualification source sha")
    projection = compute_canonical_evidence_projection(ctx)
    for filename, payload in projection.items():
        if filename.startswith("_"):
            continue
        (ctx.run_root / filename).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="EC-W3-12 reconciliation emitter")
    parser.add_argument("--source-root", type=Path, default=None)
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
    parser.add_argument("--qualification-sha", type=str, default=None)
    args = parser.parse_args()
    ctx = resolve_execution_context(
        source_root=args.source_root,
        run_root=args.run_root,
        qualification_sha=args.qualification_sha,
    )
    subprocess.run(
        [
            sys.executable,
            str(ctx.run_root / "validate_reconciliation.py"),
            "--pre-check",
            "--run-root",
            str(ctx.run_root),
        ],
        check=False,
    )
    emit(ctx)
    proc = subprocess.run(
        [
            sys.executable,
            str(ctx.run_root / "validate_reconciliation.py"),
            "--mode",
            "PARTIAL_VALIDATION",
            "--run-root",
            str(ctx.run_root),
        ],
        check=False,
    )
    if proc.returncode != 0:
        _die("validator failed after reconciliation")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
