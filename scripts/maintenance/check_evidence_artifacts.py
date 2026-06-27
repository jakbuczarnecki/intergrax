#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

"""Validate expected proof-path evidence artifacts and README proof-path references."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_EXPECTED_ARTIFACTS: tuple[str, ...] = (
    "build/evidence/core_certification/report.json",
    "build/evidence/core_certification/report.md",
    "build/evidence/trace/timeline.json",
    "build/evidence/trace/timeline.md",
    "build/evidence/live_core_probes/live_core_report.json",
    "build/evidence/live_core_probes/live_core_report.md",
    "build/evidence/eval/report.json",
    "build/evidence/eval/report.md",
    "build/evidence/cost/report.json",
    "build/evidence/cost/report.md",
    "build/evidence/posture/posture.json",
    "build/evidence/posture/posture.md",
)

_PROOF_PATH_COMMANDS: tuple[str, ...] = (
    "uv run intergrax certify core --level L2",
    "uv run intergrax trace export",
    "uv run intergrax evidence live-core",
    "uv run intergrax evidence eval",
    "uv run intergrax evidence cost",
    "uv run intergrax evidence posture",
    "uv run intergrax evidence posture export",
)

_BOUNDARY_PHRASES: tuple[str, ...] = (
    "production runtime certification",
    "security/compliance attestation",
    "real provider execution",
    "real LLM evaluation",
    "billing",
    "provider pricing",
    "cloud cost estimation",
    "product-specific acceptance",
)

_ARCH_LINK = "docs/architecture/satellites/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE_production_gates.md"
_HEP_LINK = "docs/plan/HARNESS_EVIDENCE_PACK.md"


def _repo_root(explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit.resolve()
    return Path(__file__).resolve().parents[2]


def _check_artifacts(root: Path) -> tuple[list[str], int]:
    missing: list[str] = []
    present = 0
    for rel in _EXPECTED_ARTIFACTS:
        if (root / rel).is_file():
            present += 1
        else:
            missing.append(rel)
    return missing, present


def _check_readme(root: Path) -> tuple[list[str], bool, bool, bool]:
    readme_path = root / "README.md"
    missing: list[str] = []
    if not readme_path.is_file():
        missing.append("README.md (missing file)")
        return missing, False, False, False

    text = readme_path.read_text(encoding="utf-8")

    proof_path_ok = True
    for cmd in _PROOF_PATH_COMMANDS:
        if cmd not in text:
            proof_path_ok = False
            missing.append(f"README proof path command: {cmd}")

    boundaries_ok = True
    for phrase in _BOUNDARY_PHRASES:
        if phrase not in text:
            boundaries_ok = False
            missing.append(f"README boundary phrase: {phrase}")

    links_ok = True
    if _ARCH_LINK not in text:
        links_ok = False
        missing.append(f"README link: {_ARCH_LINK}")
    if _HEP_LINK not in text:
        links_ok = False
        missing.append(f"README link: {_HEP_LINK}")

    return missing, proof_path_ok, boundaries_ok, links_ok


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate proof-path evidence artifacts and README references.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Repository root (default: inferred from script location)",
    )
    args = parser.parse_args(argv)

    root = _repo_root(args.root)
    artifact_missing, artifact_present = _check_artifacts(root)
    readme_missing, proof_path_ok, boundaries_ok, links_ok = _check_readme(root)

    artifacts_ok = not artifact_missing
    readme_ok = proof_path_ok and boundaries_ok and links_ok
    passed = artifacts_ok and readme_ok

    if passed:
        print("Evidence artifact sanity check: PASS")
        print(f"Artifacts: {artifact_present}/{len(_EXPECTED_ARTIFACTS)}")
        print("README proof path: PASS")
        print("README boundaries: PASS")
        print("README links: PASS")
        return 0

    print("Evidence artifact sanity check: FAIL")
    if artifact_missing:
        print("Missing artifacts:")
        for rel in artifact_missing:
            print(f"  - {rel}")
    readme_only = [
        item
        for item in readme_missing
        if not item.startswith("build/evidence/")
    ]
    if readme_only:
        print("Missing README entries:")
        for item in readme_only:
            print(f"  - {item}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
