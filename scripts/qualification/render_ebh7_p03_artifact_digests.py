# © Artur Czarnecki. All rights reserved.

"""Write artifact SHA256 digests for P03 independent replay."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_REPLAY = _REPO / "docs/project/maintainers/qualification/ebh_7_p03_independent_replay"
_ARTIFACTS = (
    "../EBH_7_FINAL_AUDIT_MATRICES_APPENDIX.md",
    "../EBH_7_ECAP_EXECUTABLE_CHECKLIST.md",
    "../EBH_7_PRE_CERT_INDEPENDENT_AUDIT_WORKSHEET.md",
    "../EBH_7_FINAL_CLOSURE_TRACEABILITY.md",
    "../EBH_7_INT2E_BUNDLE_QUALIFICATION_MATRIX.md",
    "../EBH_7_B6_BLOCKER_REGISTER.md",
    "../EBH_7_ONE_PASS_CLOSURE_MATRIX.md",
    "../EBH_7_RPL_FINAL_CONVERGENCE_DECISION_MATRIX.md",
    "../EBH_7_ENTERPRISE_FAMILY_COVERAGE_MATRIX.md",
    "B6_EVIDENCE_CONSISTENCY_MATRIX.md",
    "EBH_7_CLOSURE_EVIDENCE_AUDIT_MATRIX.md",
    "EBH_7_DUAL_SLUG_RPL_EF_MATRIX.md",
    "MANIFEST.json",
)


def _digest_git_path(git_path: str, head: str) -> str:
    data = subprocess.check_output(["git", "show", f"{head}:{git_path}"], cwd=_REPO)
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def _artifact_git_path(rel: str) -> str:
    rel_norm = rel.replace("\\", "/")
    if rel_norm.startswith("../"):
        return str((_REPLAY.parent / rel_norm[3:]).relative_to(_REPO)).replace("\\", "/")
    return str((_REPLAY / rel_norm).relative_to(_REPO)).replace("\\", "/")


def main() -> None:
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=_REPO, text=True).strip()
    digests: dict[str, str] = {}
    for rel in _ARTIFACTS:
        git_path = _artifact_git_path(rel)
        digests[rel.replace("\\", "/")] = _digest_git_path(git_path, head)
    out = _REPLAY / "ARTIFACT_DIGESTS.json"
    payload = {"HEAD": head, "sha256": digests}
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out.relative_to(_REPO)} ({len(digests)} artifacts)")


if __name__ == "__main__":
    main()
