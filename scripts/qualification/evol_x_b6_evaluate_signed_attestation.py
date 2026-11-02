#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""Read-only evaluation of a signed B6 attestation (external attestor / reviewer use).

Does not mint keys, sign attestations, append replay ledgers, or publish
``B6_ACCEPTANCE_PUBLICATION_SHA``. Intended for custody-separated environments
that hold trust-root public keys and optional replay-ledger snapshots only.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from scripts.qualification.evol_x_b6_attestation import (
    default_governance_path,
    evaluate_acceptance,
    load_governance,
)

_REPO = Path(__file__).resolve().parents[2]


def _load_json_object(path: Path) -> dict[str, object]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must be a JSON object")
    return data


def _load_replay_ledger(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or any(not isinstance(item, str) for item in data):
        raise ValueError("replay ledger must be a JSON array of strings")
    return set(data)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", type=Path, default=_REPO)
    parser.add_argument("--governance", type=Path, default=None)
    parser.add_argument("--attestation", type=Path, required=True)
    parser.add_argument("--trust-root", type=Path, required=True)
    parser.add_argument("--replay-ledger", type=Path, default=None)
    parser.add_argument("--mode-a", choices=("PASS", "FAIL"), default="PASS")
    parser.add_argument("--mode-b", choices=("PASS", "FAIL", "SKIPPED"), default="PASS")
    parser.add_argument("--technical", choices=("PASS", "FAIL", "INCOMPLETE", "NOT_RUN"), default="PASS")
    parser.add_argument(
        "--tool-digest",
        default=None,
        help="Expected sha256 of evol_x_b6_independent_verification.py (hex, 64 chars)",
    )
    parser.add_argument("--output", type=Path, default=None, help="Write immutable result JSON here")
    args = parser.parse_args(argv)

    governance_path = args.governance or default_governance_path(args.repo)
    governance = load_governance(governance_path)
    attestation = _load_json_object(args.attestation)
    trust_root = _load_json_object(args.trust_root)
    replay_ledger = _load_replay_ledger(args.replay_ledger)
    tool_digest = args.tool_digest
    if tool_digest is None:
        verifier = args.repo / "scripts/qualification/evol_x_b6_independent_verification.py"
        import hashlib

        h = hashlib.sha256()
        with verifier.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                h.update(chunk)
        tool_digest = h.hexdigest()

    mode_b = args.mode_b
    if mode_b == "SKIPPED":
        technical = "INCOMPLETE" if args.technical == "PASS" else args.technical
    else:
        technical = args.technical

    result = evaluate_acceptance(
        technical_verdict=technical,
        mode_a=args.mode_a,
        mode_b=mode_b,
        governance=governance,
        attestation=attestation,
        trust_root=trust_root,
        observed={"mode_a": args.mode_a, "mode_b": mode_b},
        expected_tool_digest=tool_digest,
        replay_ledger=replay_ledger,
        now=datetime.now(UTC),
    )
    result["b6_acceptance_publication_sha"] = None
    result["evol_x_closed"] = False
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.write_text(payload, encoding="utf-8")
    sys.stdout.write(payload)
    return 0 if result["formal_acceptance_eligibility"] == "ELIGIBLE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
