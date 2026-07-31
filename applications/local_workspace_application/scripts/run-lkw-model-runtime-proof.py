#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""Operator entrypoint for LKW-MODEL-RUNTIME-1 portability proof."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

_SCRIPT_PATH = Path(__file__).resolve()
_APP_DIR = _SCRIPT_PATH.parent.parent
_EVIDENCE_DIR = _APP_DIR / "docs" / "evidence"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write-evidence",
        action="store_true",
        help="Write sanitized JSON/Markdown evidence when proof passes.",
    )
    args = parser.parse_args()

    from local_workspace_application.model_runtime_proof.config import (
        load_proof_config_from_env,
    )
    from local_workspace_application.model_runtime_proof.runner import (
        run_model_runtime_proof,
    )
    from local_workspace_application.model_runtime_proof.report import (
        stale_evidence_notice,
    )

    config = load_proof_config_from_env()
    json_path = _EVIDENCE_DIR / "LKW_MODEL_RUNTIME_PORTABILITY.json"
    markdown_path = _EVIDENCE_DIR / "LKW_MODEL_RUNTIME_PORTABILITY.md"
    result = asyncio.run(
        run_model_runtime_proof(
            config,
            evidence_json=json_path if args.write_evidence else None,
            evidence_markdown=markdown_path if args.write_evidence else None,
        )
    )
    if not args.write_evidence:
        stale = stale_evidence_notice(json_path)
        if stale:
            print(stale)
    if args.write_evidence and result.overall_status.value != "PASS":
        return 1
    return 0 if result.overall_status.value == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
