#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Read-only review CLI for token regression diagnostic artifact folders."""

from __future__ import annotations

import argparse
import json
import sys

from intergrax.runtime.token_optimization.regression_artifact_review import (
    format_token_regression_artifact_review,
    review_token_regression_artifacts,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Review an existing token regression diagnostic artifact folder "
            "and emit a human-readable interpretation."
        ),
    )
    parser.add_argument(
        "artifact_dir",
        help="Path to a diagnostic artifact directory produced by check_token_regression_benchmarks.py.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable review JSON on stdout.",
    )
    args = parser.parse_args(argv)

    review = review_token_regression_artifacts(args.artifact_dir)

    if args.json:
        print(json.dumps(review, indent=2, sort_keys=True))
    else:
        print(format_token_regression_artifact_review(review), end="")

    if review["status"] == "fail":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
