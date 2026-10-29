# © Artur Czarnecki. All rights reserved.

"""Publish EBH-6 Protocol 03 independent replay bundle under docs/ (auditor-accessible)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_OUT = _REPO / "docs/project/maintainers/qualification/ebh_6_p03_independent_replay"


def _git_sha() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        cwd=_REPO,
    )
    return result.stdout.strip()


def main() -> int:
    sys.path.insert(0, str(_REPO))
    from tests.qualification.ebh_6._evidence_manifest import (
        EBH6_CERTIFICATION_HANDOFF_SHA,
        EBH6_PUBLICATION_SHA,
    )
    from tests.qualification.ebh_6._p03_handoff_export import p03_handoff_bundle
    from tests.qualification.ebh_6._pre_cert_evidence_export import pre_cert_export_dict

    sha = _git_sha()
    _OUT.mkdir(parents=True, exist_ok=True)
    handoff = p03_handoff_bundle()
    (_OUT / "p03-handoff.json").write_text(json.dumps(handoff, indent=2, sort_keys=True), encoding="utf-8")
    (_OUT / "pre-cert-export.json").write_text(
        json.dumps(pre_cert_export_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    manifest = {
        "TEST_EXECUTION_SHA": sha,
        "CERTIFICATION_HANDOFF_SUBSTANCE_SHA": EBH6_CERTIFICATION_HANDOFF_SHA,
        "EBH6_PUBLICATION_SHA": EBH6_PUBLICATION_SHA,
        "pytest_suite": "tests/qualification/ebh_6/",
        "aj_group_count": handoff["aj_group_count"],
        "family_count": handoff["family_count"],
        "l9_blocker_count": handoff["l9_blocker_count"],
        "certification_banner": handoff["certification_banner"],
        "independent_acceptance": "PENDING — no Protocol 03 PASS record",
        "artifacts": {
            "p03_handoff": "p03-handoff.json",
            "pre_cert_export": "pre-cert-export.json",
            "pytest_transcript": "pytest-transcript.txt",
        },
    }
    (_OUT / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote EBH-6 P03 independent replay bundle @ {sha} -> {_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
