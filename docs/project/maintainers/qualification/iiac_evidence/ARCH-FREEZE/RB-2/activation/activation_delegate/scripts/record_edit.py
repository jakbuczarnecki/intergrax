"""Edit IIAC_POLICY_ACTIVATION_RECORD.json for RB-2 dev-main rebind."""
import json
import sys
from pathlib import Path

Q = "docs/project/maintainers/qualification/"
RECORD = Path(Q + "IIAC_POLICY_ACTIVATION_RECORD.json")
G = "9d3a042fcb7219ab344e526e036ae3e3ca3dde9e"
E = sys.argv[3] if len(sys.argv) > 3 else "__EVIDENCE_SHA__"
C = sys.argv[4] if len(sys.argv) > 4 else E
MANIFEST = Q + "iiac_evidence/ARCH-FREEZE/RB-2/activation/iiac_audit_session_manifest.IIAC-ACT-N11-ARCH-FREEZE-RB2-DEV-MAIN.json"
HANDOFF = Q + "iiac_evidence/ARCH-FREEZE/RB-2/activation/IIAC_SESSION_HANDOFF_RECORD_ARCH_FREEZE_RB2_DEV_MAIN_ACTIVATION.md"
OD_PATH = Q + "IIAC_OD_ACTIVATION_ARCH_FREEZE_RB2_DEV_MAIN_2026-11-23.md"
OD_ID = "IIAC-OD-ARCH-FREEZE-RB2-DEV-MAIN-2026-11-23"
OD_LINE = "Approved IIAC policy activation IIAC-2026-11-05.1: operator @ 2026-11-23T15:50:00+01:00"
REPORT_DIGEST = "c405c65faf53243a0511b0b9584a9f3abda4db08c09ee9b9266c86fe9f6b7a4e"


def digest_file(rel: str) -> str:
    import hashlib
    return hashlib.sha256(Path(rel).read_bytes()).hexdigest()


def write(doc: dict) -> None:
    RECORD.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")


def main() -> int:
    if len(sys.argv) < 2:
        return 2
    doc = json.loads(RECORD.read_text(encoding="utf-8"))
    if sys.argv[1] == "phase1":
        doc["effective_from_sha"] = ""
        doc["policy_activation_sha"] = ""
        doc["corrective_implementation_sha"] = G
        doc["audit_execution_sha"] = E
        doc["evidence_publication_sha"] = E
        doc["certification_publication_sha"] = C
        doc["session_b_verdict"] = "PASS"
        doc["session_b_governance_audit_report_digest"] = REPORT_DIGEST
        doc["session_b_manifest_path"] = MANIFEST
        doc["session_b_manifest_digest"] = digest_file(MANIFEST)
        doc["handoff_record_path"] = HANDOFF
        doc["handoff_digest"] = digest_file(HANDOFF)
        doc["operator_decision_path"] = OD_PATH
        doc["operator_decision_id"] = OD_ID
        doc["operator_activation_line"] = OD_LINE
        doc["operator_decision_class"] = "OD-ARCH-FREEZE-RB2-DEV-MAIN"
        doc["notes"] = (
            "Supplemental ACT-N11 rebind of development for PR #744 at product tree G=9d3a042, "
            f"evidence and audit execution E={E}, certification publication C={C}. "
            "Delegate DELEGATION-ARCH-FREEZE-RB2-DEV-MAIN-2026-11-23; supplemental operator chat auth 2026-11-23."
        )
    elif sys.argv[1] == "phase2":
        sha = sys.argv[2]
        doc["effective_from_sha"] = sha
        doc["policy_activation_sha"] = sha
    else:
        return 2
    write(doc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())