#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""EVOL-X Protocol 03 evidence verifier (stage-local; does not modify EBH-7 tooling)."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-execution-sha", required=True)
    args = parser.parse_args(argv)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=_REPO, text=True).strip()
    if head != args.test_execution_sha:
        sys.stderr.write(f"HEAD {head} != TEST_EXECUTION_SHA {args.test_execution_sha}\n")
        return 2
    payload = {"status": "EVIDENCE_INTEGRITY_OK", "test_execution_sha": head}
    sys.stdout.write(json.dumps(payload) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
