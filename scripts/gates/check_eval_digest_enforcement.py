#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""T-AF2-RW-GATES: enforce eval run and comparison digests on promotion evidence paths."""

from __future__ import annotations

import argparse
import json
import sys

from intergrax.contracts.evaluation_contracts import (
    EvalComparison,
    EvalDigestMismatchError,
    EvalRun,
    enforce_eval_run_digests,
    verify_eval_comparison_digests,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=argparse.FileType("r", encoding="utf-8"), help="EvalRun JSON")
    parser.add_argument("--declared-inputs-digest", required=False)
    parser.add_argument("--declared-result-digest", required=False)
    parser.add_argument("--comparison", type=argparse.FileType("r", encoding="utf-8"))
    parser.add_argument("--baseline-run", type=argparse.FileType("r", encoding="utf-8"))
    parser.add_argument("--candidate-run", type=argparse.FileType("r", encoding="utf-8"))
    args = parser.parse_args()

    try:
        if args.run is not None:
            run = EvalRun.model_validate(json.load(args.run))
            if not args.declared_inputs_digest or not args.declared_result_digest:
                print("DENIED: run mode requires declared digests", file=sys.stderr)
                return 1
            enforce_eval_run_digests(
                run,
                declared_inputs_digest=args.declared_inputs_digest,
                declared_result_digest=args.declared_result_digest,
            )
        if args.comparison is not None:
            comparison = EvalComparison.model_validate(json.load(args.comparison))
            if args.baseline_run is None or args.candidate_run is None:
                print("DENIED: comparison mode requires baseline and candidate runs", file=sys.stderr)
                return 1
            baseline = EvalRun.model_validate(json.load(args.baseline_run))
            candidate = EvalRun.model_validate(json.load(args.candidate_run))
            verify_eval_comparison_digests(comparison, baseline_run=baseline, candidate_run=candidate)
    except EvalDigestMismatchError as exc:
        print(f"DENIED: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 - gate must deny unknown failures
        print(f"DENIED: {exc}", file=sys.stderr)
        return 1

    print("OK: eval digest enforcement")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
