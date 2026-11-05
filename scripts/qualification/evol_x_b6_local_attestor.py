#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""Local read-only B6 attestor process (separate from implementer signing).

Runs mechanical verification in a subprocess, builds an **unsigned** attestation
payload, and writes it to a session path. Does not hold private keys and does
not set ``B6_ACCEPTANCE_PUBLICATION_SHA``.

Independence requires a **separate OS principal** running this script against a
read-only clone; two Cursor sessions are not sufficient (ECAP-P4).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path

from scripts.qualification.evol_x_b6_attestation import (
    IMPLEMENTER_IDENTITIES,
    build_unsigned_attestation_payload,
)

_REPO = Path(__file__).resolve().parents[2]
_VERIFIER = _REPO / "scripts" / "qualification" / "evol_x_b6_independent_verification.py"


def _tool_digest() -> str:
    h = hashlib.sha256()
    with _VERIFIER.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_report(path: Path) -> dict[str, object]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("verification report must be a JSON object")
    return data


def _technical_from_report(report: dict[str, object]) -> tuple[str, str, str]:
    tech = report.get("technical_verification")
    if not isinstance(tech, dict):
        return "FAIL", "FAIL", "FAIL"
    verdict = str(tech.get("verdict", "FAIL"))
    auto = report.get("automation_qualification")
    if isinstance(auto, dict):
        mode_a = str(auto.get("mode_a", "FAIL"))
        mode_b = str(auto.get("mode_b", "FAIL"))
    else:
        mode_a = mode_b = "FAIL"
    return verdict, mode_a, mode_b


def _assert_readonly_attestor_env() -> None:
    if os.environ.get("EVOL_B6_ATTESTOR_READONLY") != "1":
        raise PermissionError("set EVOL_B6_ATTESTOR_READONLY=1 for attestor process (separate principal recommended)")
    identity = os.environ.get("EVOL_B6_ATTESTOR_IDENTITY", "").strip().lower()
    if not identity or identity in IMPLEMENTER_IDENTITIES:
        raise PermissionError("EVOL_B6_ATTESTOR_IDENTITY must name a non-implementer attestor")


def run_verification(
    *,
    repo: Path,
    session_root: Path,
    skip_mode_b: bool,
    adversarial: bool,
) -> Path:
    session_root.mkdir(parents=True, exist_ok=True)
    report_path = session_root / "independent_verification_report.json"
    cmd = [
        sys.executable,
        str(_VERIFIER),
        "--repo",
        str(repo),
        "--session-root",
        str(session_root),
        "--report-json",
        str(report_path),
    ]
    if skip_mode_b:
        cmd.append("--skip-mode-b")
    if adversarial:
        cmd.append("--adversarial-invalid-ref")
    proc = subprocess.run(cmd, cwd=repo, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        raise RuntimeError(f"verification subprocess exit {proc.returncode}")
    return report_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", type=Path, default=_REPO)
    parser.add_argument("--session-root", type=Path, default=None)
    parser.add_argument("--verification-report", type=Path, default=None, help="Reuse existing report JSON")
    parser.add_argument("--skip-mode-b", action="store_true")
    parser.add_argument("--adversarial-invalid-ref", action="store_true")
    parser.add_argument("--output", type=Path, required=True, help="Unsigned attestation JSON path")
    parser.add_argument("--trust-root-ref", required=True)
    parser.add_argument("--key-id", required=True)
    parser.add_argument("--evidence-producer", default="evol-x-evidence-publisher")
    parser.add_argument("--credential-class", default="PRODUCTION_INDEPENDENT")
    parser.add_argument(
        "--formal-disposition",
        default="ACCEPTED_AUTOMATED_B6",
        choices=("NOT_ACCEPTED", "ACCEPTED_AUTOMATED_B6"),
    )
    args = parser.parse_args(argv)

    try:
        _assert_readonly_attestor_env()
    except PermissionError as exc:
        sys.stderr.write(f"{exc}\n")
        return 1

    attestor = os.environ["EVOL_B6_ATTESTOR_IDENTITY"].strip()
    try:
        if args.verification_report is not None:
            report = _load_report(args.verification_report)
        else:
            if args.session_root is None:
                raise ValueError("--session-root is required when not using --verification-report")
            report_path = run_verification(
                repo=args.repo,
                session_root=args.session_root,
                skip_mode_b=args.skip_mode_b,
                adversarial=args.adversarial_invalid_ref,
            )
            report = _load_report(report_path)
    except (OSError, RuntimeError, ValueError) as exc:
        sys.stderr.write(f"attestor verification error: {exc}\n")
        return 1

    verdict, mode_a, mode_b = _technical_from_report(report)
    if verdict != "PASS" or mode_a != "PASS":
        sys.stderr.write(f"technical verification not PASS: {verdict} mode_a={mode_a} mode_b={mode_b}\n")
        return 2

    now = datetime.now(UTC)
    payload = build_unsigned_attestation_payload(
        verification_tool_digest=_tool_digest(),
        attestor_identity=attestor,
        evidence_producer_identity=args.evidence_producer,
        key_id=args.key_id,
        trust_root_ref=args.trust_root_ref,
        attestation_id=f"evol-b6-{uuid.uuid4()}",
        credential_class=args.credential_class,
        formal_acceptance_disposition=args.formal_disposition,
        mode_a_status=mode_a,
        mode_b_status=mode_b if mode_b in {"PASS", "FAIL", "SKIPPED"} else "FAIL",
        now=now,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    sys.stdout.write(json.dumps({"status": "UNSIGNED_ATTESTATION_READY", "path": str(args.output)}) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
