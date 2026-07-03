#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Local gate for deterministic token optimization regression benchmarks."""

from __future__ import annotations

import argparse
import json

from intergrax.runtime.token_optimization.fixture_dataset import (
    load_token_regression_fixture_dataset,
)
from intergrax.runtime.token_optimization.regression import (
    format_regression_summary,
    regression_summary_to_dict,
    run_token_regression_benchmarks,
)
from intergrax.runtime.token_optimization.regression_gate import (
    TokenRegressionGateStatus,
    TokenRegressionGateThresholds,
    evaluate_token_regression_gate,
    format_token_regression_gate,
    token_regression_gate_to_dict,
)
from intergrax.runtime.token_optimization.regression_report import (
    build_token_regression_report,
    format_token_regression_report,
    token_regression_report_to_dict,
)


def _build_gate_thresholds(args: argparse.Namespace) -> TokenRegressionGateThresholds | None:
    if args.min_total_saved_ratio is None and args.min_total_saved_tokens is None:
        return None
    return TokenRegressionGateThresholds(
        min_total_saved_ratio=args.min_total_saved_ratio,
        min_total_saved_tokens=args.min_total_saved_tokens,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run deterministic token optimization regression benchmarks.",
    )
    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON summary on stdout.",
    )
    output_group.add_argument(
        "--report",
        action="store_true",
        help="Emit a redaction-safe human-readable regression report on stdout.",
    )
    output_group.add_argument(
        "--report-json",
        action="store_true",
        help="Emit a redaction-safe regression report as JSON on stdout.",
    )
    output_group.add_argument(
        "--gate",
        action="store_true",
        help="Emit a human-readable regression gate summary on stdout.",
    )
    output_group.add_argument(
        "--gate-json",
        action="store_true",
        help="Emit a regression gate result as JSON on stdout.",
    )
    parser.add_argument(
        "--min-total-saved-ratio",
        type=float,
        default=None,
        metavar="FLOAT",
        help="Gate-only: require aggregate saved-token ratio to be at least FLOAT.",
    )
    parser.add_argument(
        "--min-total-saved-tokens",
        type=int,
        default=None,
        metavar="INT",
        help="Gate-only: require aggregate saved tokens to be at least INT.",
    )
    parser.add_argument(
        "--fixture-dataset",
        default=None,
        metavar="PATH",
        help="Load regression fixtures from a file-backed dataset directory.",
    )
    args = parser.parse_args(argv)

    fixtures = None
    if args.fixture_dataset is not None:
        dataset = load_token_regression_fixture_dataset(args.fixture_dataset)
        fixtures = dataset.fixtures

    summary = run_token_regression_benchmarks(fixtures=fixtures)
    benchmark_failed = summary.failed > 0

    if args.report:
        report = build_token_regression_report(summary)
        print(format_token_regression_report(report))
    elif args.report_json:
        report = build_token_regression_report(summary)
        print(
            json.dumps(
                token_regression_report_to_dict(report),
                indent=2,
                sort_keys=True,
            )
        )
    elif args.gate:
        report = build_token_regression_report(summary)
        gate = evaluate_token_regression_gate(
            summary,
            thresholds=_build_gate_thresholds(args),
            report=report,
        )
        print(format_token_regression_gate(gate))
        if gate.status != TokenRegressionGateStatus.PASS:
            return 1
    elif args.gate_json:
        report = build_token_regression_report(summary)
        gate = evaluate_token_regression_gate(
            summary,
            thresholds=_build_gate_thresholds(args),
            report=report,
        )
        print(
            json.dumps(
                token_regression_gate_to_dict(gate),
                indent=2,
                sort_keys=True,
            )
        )
        if gate.status != TokenRegressionGateStatus.PASS:
            return 1
    elif args.json:
        print(json.dumps(regression_summary_to_dict(summary), indent=2, sort_keys=True))
    else:
        print(format_regression_summary(summary))

    return 0 if not benchmark_failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
