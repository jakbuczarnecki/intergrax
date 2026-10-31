# © Artur Czarnecki. All rights reserved.

"""Post-run F10 execution attestation: every PROFILE_BOUND evidence node passed in JUnit @ executed SHA."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))

from tests.qualification.ebh_7._ebh7_f10_ecap_normative_matrix import attest_junit, certify_f10  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--junit", required=True, type=Path)
    parser.add_argument("--test-execution-sha", required=True)
    parser.add_argument("--replay-head-sha", required=True)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)

    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=_REPO, text=True).strip()
    cert = certify_f10()
    problems: list[str] = []
    if head != args.replay_head_sha:
        problems.append(f"executed HEAD {head} != replay HEAD {args.replay_head_sha}")
    if cert.test_execution_sha != args.test_execution_sha:
        problems.append(f"MANIFEST TEST_EXECUTION_SHA {cert.test_execution_sha} != {args.test_execution_sha}")
    if not args.junit.is_file():
        problems.append(f"JUnit missing: {args.junit}")
        nodes, missing = {}, ()
    else:
        nodes, missing = attest_junit(args.junit, cert)
    problems.extend(f"evidence not passed in JUnit: {node}" for node in missing)
    payload = {
        "protocol": "EBH-7-B6-F10-EXECUTION-ATTESTATION",
        "executed_head_sha": head,
        "test_execution_sha": cert.test_execution_sha,
        "computed_disposition": cert.disposition,
        "verdicts": {v.gate.value: v.verdict.value for v in cert.verdicts},
        "failed_obligations": list(cert.failed_obligation_ids()),
        "evidence_nodes": nodes,
        "problems": problems,
        "status": "PASS" if not problems and not cert.failed_obligation_ids() else "FAIL",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"f10_attestation={payload['status']} nodes={len(nodes)} problems={len(problems)}")
    for problem in problems[:20]:
        print(f"  {problem}")
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
