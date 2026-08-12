#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""Canonical Intergrax proof suite entrypoint (PUBLIC-PROOF-GATE-1)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from scripts.proof.intergrax_proof_contracts import ProofProfile
from scripts.proof.intergrax_proof_runner import (
    RunnerConfig,
    render_console_summary,
    run_suite,
    suite_exit_code,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the canonical Intergrax proof suite for this commit.",
    )
    parser.add_argument(
        "--profile",
        required=True,
        choices=[item.value for item in ProofProfile],
        help="quick: fast local proofs; full: all local proofs; live: adds external providers",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-proof diagnostics on the console.",
    )
    parser.add_argument(
        "--allow-external-mutating",
        action="store_true",
        help="Opt in to external mutating proofs registered in the manifest.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve manifest selection only; do not execute child proofs.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    config = RunnerConfig(
        profile=ProofProfile(args.profile),
        repo_root=_REPO_ROOT,
        verbose=bool(args.verbose),
        allow_external_mutating=bool(args.allow_external_mutating),
        dry_run=bool(args.dry_run),
    )

    receipt, receipt_path = run_suite(config)
    print(render_console_summary(receipt, verbose=config.verbose))
    if receipt_path is not None:
        print(f"receipt: {receipt_path.relative_to(_REPO_ROOT)}")
    return suite_exit_code(receipt)


if __name__ == "__main__":
    sys.exit(main())
