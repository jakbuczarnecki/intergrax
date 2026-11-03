#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""L1 → L2 → semantic 85 → L3 certifying orchestrator (ENT-AUDIT-X C051)."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from scripts.enterprise_audit import qual_wave_lib as qw

_DEFAULT_LOG_ROOT = Path(".tmp/session/ent-audit-x-qual-wave")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser(description="ENT-AUDIT-X qualification wave orchestrator")
    parser.add_argument(
        "--profile",
        choices=("pre-qual", "certifying"),
        default="pre-qual",
        help="pre-qual runs L1/L2/semantic gates; certifying adds L3 FULL + first FREEZE",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print planned stages only")
    parser.add_argument("--log-dir", type=Path, default=_DEFAULT_LOG_ROOT)
    parser.add_argument("--l3-output", type=Path, default=None)
    parser.add_argument("--wave-output", type=Path, default=None)
    parser.add_argument(
        "--inject-fail-after",
        choices=(
            "l1",
            "l2",
            "semantic_acceptance",
            "semantic_matrix",
            "adv31",
            "ten13",
            "vbn9",
            "d31",
        ),
        help="Self-test hook: exit 2 when named stage would pass alone",
    )
    args = parser.parse_args()
    root = repo_root()

    if args.dry_run:
        print("profile=", args.profile)
        for line in (
            "l1 ent_l1",
            "l2 ent_l2 integration/",
            "semantic_acceptance ent_semantic_gate ENT-A x63 ENT_AUDIT_X_SEMANTIC=1",
            "semantic_matrix ent_semantic_matrix x85 ENT_AUDIT_X_GATE_MATRIX=1",
            "adv31 adversarial ADV01..31",
            "ten13 TEN acceptance x13",
            "vbn9 VBN acceptance x9",
            "d31 vbn domain validator D01-D31",
            "l3_cli_full CLI FULL certifying only",
            "first_freeze_candidate certifying only",
        ):
            print("stage", line)
        return 0

    if args.inject_fail_after:
        mapping: dict[str, tuple[str | None, list[str], dict[str, str] | None]] = {
            "l1": ("ent_l1", ["tests/qualification/ent_audit_x/unit/test_task_id_registry.py"], None),
            "l2": ("ent_l2", ["tests/qualification/ent_audit_x/integration/test_adversarial_adv01_30.py"], None),
            "semantic_acceptance": (
                "ent_semantic_gate",
                qw._glob_paths(root, qw._ENT_A_ACCEPTANCE_GLOB),
                {"ENT_AUDIT_X_SEMANTIC": "1"},
            ),
            "semantic_matrix": (
                "ent_semantic_matrix",
                ["tests/qualification/ent_audit_x/integration/test_semantic_gate_execution.py"],
                {"ENT_AUDIT_X_GATE_MATRIX": "1"},
            ),
            "adv31": (None, list(qw._ADV_PATHS), None),
            "ten13": (
                "ent_semantic_gate",
                qw._glob_paths(root, qw._TEN_ACCEPTANCE_GLOB),
                {"ENT_AUDIT_X_SEMANTIC": "1"},
            ),
            "vbn9": (
                "ent_semantic_gate",
                qw._glob_paths(root, qw._VBN_ACCEPTANCE_GLOB),
                {"ENT_AUDIT_X_SEMANTIC": "1"},
            ),
            "d31": (None, list(qw._D31_PATHS), None),
        }
        marker, paths, env = mapping[args.inject_fail_after]
        code, _, _ = qw.run_pytest_strict(root, marker=marker, paths=paths, env=env, timeout_s=7200)
        return 2 if code == 0 else 0

    pinned = os.environ.get("TEST_EXECUTION_SHA")
    l3_out = args.l3_output or (args.log_dir / "l3-cli-full")
    wave_out = args.wave_output or (args.log_dir / "wave-artifacts")
    report = qw.run_qualification_wave(
        root,
        profile=args.profile,
        pinned_sha=pinned,
        l3_output=l3_out,
        wave_output=wave_out,
    )
    qw.write_report(report, args.log_dir)
    if report.status not in ("PASS", "PRE_QUAL_PASS"):
        print(report.error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
