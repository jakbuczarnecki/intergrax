"""Edit IIAC_POLICY_ACTIVATION_RECORD.json for the PDC-X P8-04 rebind (activation delegate).

Usage (from the checkout root):
  python record_edit.py phase1
  python record_edit.py phase2 <phase-1-commit-sha>
Writes JSON with indent=2, ensure_ascii=False, a trailing newline, UTF-8, LF (the record path is not -text).
"""
import json
import sys
from pathlib import Path

Q = "docs/project/maintainers/qualification/"
RECORD = Path(Q + "IIAC_POLICY_ACTIVATION_RECORD.json")
G = "969cbd964c679d563b02e55f065b452d46fdf90b"
E = "a73d15ab341b30e9748aad03894f6faa75ad4e09"
C = "fd2ad28efde6216be38194b020188d116264adb6"
OD_PATH = Q + "IIAC_OD_ACTIVATION_PDC_X_P8_2026-11-16.md"
OD_ID = "IIAC-OD-PDC-X-P8-ACT-N11-2026-11-16"
OD_LINE = "Approved IIAC policy activation IIAC-2026-11-05.1: operator @ 2026-11-16T09:06:54+01:00"
MANIFEST = Q + "iiac_evidence/PDC-X/P8/activation/iiac_audit_session_manifest.IIAC-ACT-N11-PDC-X-P8-04.json"
MANIFEST_DIGEST = "77fa2babd02df6fbbd2e0d5b34413a769e069ac293e485daf49ec9a779dc476f"
HANDOFF = Q + "iiac_evidence/PDC-X/P8/activation/IIAC_SESSION_HANDOFF_RECORD_PDC_X_P8_04_ACTIVATION.md"
HANDOFF_DIGEST = "a36b2ac35eaf0782f4d40ef9b76a4dd745b0c4e70d839e2df2a7fa819b1fd121"
NOTES = (
    "ACT-N11 rebind of the PDC-X product tree at the global audit SHA G=969cbd964 (74 product paths versus a1b3053b4; "
    "192 versus the P6 pin 2d3eceb6c), audit and evidence publication E=a73d15ab3 (merge of PR #597), certification "
    "publication C=fd2ad28ef. Previous bindings preserved in history: P6 activation (phase-2 19c617365), the first "
    "MCP-RC binding (phase-2 9949fe2c3) and MCP-RC re-pin 2 (phase-2 88fd679e6). The activation line is the plain form "
    "written by the operator-designated activation delegate (DELEGATION-PDC-P8-ACT.md (c)); it is not claimed to be "
    "operator-written. The bound verdict PASS is the certifier's own, authored from the PASS_SCOPED global audit "
    "(DELEGATION-PDC-P8-ACT.md (b)); scope limits and declared exceptions are in the binding manifest limitations and "
    "in the decision file. No stage closure, no FRZ promotion. effective_from_sha and policy_activation_sha are set in "
    "phase 2."
)


def write(doc: dict) -> None:
    RECORD.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")


def main() -> int:
    doc = json.loads(RECORD.read_text(encoding="utf-8"))
    if sys.argv[1] == "phase1":
        doc["effective_from_sha"] = ""
        doc["policy_activation_sha"] = ""
        doc["corrective_implementation_sha"] = G
        doc["audit_execution_sha"] = E
        doc["evidence_publication_sha"] = E
        doc["certification_publication_sha"] = C
        doc["session_b_verdict"] = "PASS"
        doc["session_b_manifest_path"] = MANIFEST
        doc["session_b_manifest_digest"] = MANIFEST_DIGEST
        doc["handoff_record_path"] = HANDOFF
        doc["handoff_digest"] = HANDOFF_DIGEST
        doc["operator_decision_path"] = OD_PATH
        doc["operator_decision_id"] = OD_ID
        doc["operator_activation_line"] = OD_LINE
        doc["operator_decision_class"] = "OD-PDC-X-P8-ACT-N11"
        doc["notes"] = NOTES
    elif sys.argv[1] == "phase2":
        sha = sys.argv[2]
        assert len(sha) == 40 and sha == sha.lower()
        doc["effective_from_sha"] = sha
        doc["policy_activation_sha"] = sha
    else:
        return 2
    write(doc)
    return 0


if __name__ == "__main__":
    sys.exit(main())
