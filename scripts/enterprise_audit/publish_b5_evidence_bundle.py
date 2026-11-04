#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""Publish ENT-AUDIT-X B5 evidence tree under tracked b6_independent_handoff (C050/C058)."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from intergrax.platform.enterprise_audit.evidence import sha256_file, verify_evidence_manifest
from intergrax.platform.enterprise_audit.validators import verify_b5_publication_tree

_HANDOFF_ROOT = Path("docs/project/maintainers/qualification/ent_audit_x/b6_independent_handoff")

_ARTIFACT_NAMES = (
    "enterprise_audit_report.json",
    "enterprise_audit_report.md",
    "enterprise_audit_outcomes.json",
    "vbn_domains.v1.json",
    "frz_coverage.v1.json",
    "evidence_manifest.v1.json",
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _git_head(root: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()


def _is_ancestor(root: Path, ancestor: str, descendant: str) -> bool:
    proc = subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, descendant],
        cwd=root,
        capture_output=True,
        check=False,
    )
    return proc.returncode == 0


def publish(
    *,
    root: Path,
    qualification_sha: str,
    l3_source: Path,
    qual_wave_source: Path | None,
    dest: Path | None,
) -> Path:
    if not _is_ancestor(root, qualification_sha, _git_head(root)):
        raise ValueError(f"qualification_sha {qualification_sha} must be ancestor of HEAD")
    if not l3_source.is_dir():
        raise ValueError(f"L3 source directory missing: {l3_source}")
    for name in _ARTIFACT_NAMES:
        if not (l3_source / name).is_file():
            raise ValueError(f"missing certifying artifact: {name}")

    out = (dest or (_HANDOFF_ROOT / f"qualification_at_{qualification_sha}")).resolve()
    root = root.resolve()
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    l3_dest = (out / "l3-cli-full").resolve()
    l3_dest.mkdir(parents=True, exist_ok=True)
    for name in _ARTIFACT_NAMES:
        shutil.copy2(l3_source / name, l3_dest / name)

    if qual_wave_source and qual_wave_source.is_file():
        shutil.copy2(qual_wave_source, out / "qual_wave_report.json")

    if not verify_evidence_manifest(l3_dest):
        raise ValueError("evidence_manifest verification failed after copy")

    index = {
        "schema_version": 1,
        "qualification_sha": qualification_sha,
        "publication_role": "EVIDENCE_CANDIDATE_TREE",
        "l3_bundle_relpath": l3_dest.relative_to(root).as_posix(),
        "artifacts": [
            {"path": (l3_dest / name).relative_to(root).as_posix(), "sha256": sha256_file(l3_dest / name)}
            for name in _ARTIFACT_NAMES
        ],
    }
    qw_dest = out / "qual_wave_report.json"
    if qw_dest.is_file():
        index["qual_wave_report"] = {
            "path": qw_dest.relative_to(root).as_posix(),
            "sha256": sha256_file(qw_dest),
        }
    index_path = out / "publication_index.v1.json"
    index_path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    pub_errors = verify_b5_publication_tree(root, out, expected_qualification_sha=qualification_sha)
    if pub_errors:
        raise ValueError("B5 publication verification failed: " + "; ".join(pub_errors[:8]))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Publish B5 ENT-AUDIT-X evidence handoff tree")
    parser.add_argument("--qualification-sha", required=True)
    parser.add_argument(
        "--l3-source",
        type=Path,
        default=None,
        help="Certifying FULL output dir (default: .tmp/session/ent-audit-x-certifying-final/l3-cli-full)",
    )
    parser.add_argument(
        "--qual-wave-source",
        type=Path,
        default=None,
        help="qual_wave_report.json (default: .tmp/session/ent-audit-x-certifying-final/qual_wave_report.json)",
    )
    parser.add_argument("--dest", type=Path, default=None)
    args = parser.parse_args()
    root = _repo_root()
    l3 = args.l3_source or (root / ".tmp/session/ent-audit-x-certifying-final/l3-cli-full")
    qw = args.qual_wave_source or (root / ".tmp/session/ent-audit-x-certifying-final/qual_wave_report.json")
    try:
        out = publish(
            root=root,
            qualification_sha=args.qualification_sha,
            l3_source=l3,
            qual_wave_source=qw if qw.is_file() else None,
            dest=args.dest,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(out.relative_to(root))
    return 0


if __name__ == "__main__":
    sys.exit(main())
