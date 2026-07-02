#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Local gate for deterministic token optimization regression benchmarks."""

from __future__ import annotations

import argparse
import json
import sys

from intergrax.runtime.token_optimization.regression import (
    format_regression_summary,
    regression_summary_to_dict,
    run_token_regression_benchmarks,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run deterministic token optimization regression benchmarks.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON summary on stdout.",
    )
    args = parser.parse_args(argv)

    summary = run_token_regression_benchmarks()

    if args.json:
        print(json.dumps(regression_summary_to_dict(summary), indent=2, sort_keys=True))
    else:
        print(format_regression_summary(summary))

    return 0 if summary.failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
