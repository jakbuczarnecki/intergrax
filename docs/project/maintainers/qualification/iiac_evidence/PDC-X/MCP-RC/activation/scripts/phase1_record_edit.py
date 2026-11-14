import json, hashlib
from pathlib import Path
R = Path("docs/project/maintainers/qualification/IIAC_POLICY_ACTIVATION_RECORD.json")
raw = R.read_bytes()
print("eol crlf:", b"\r\n" in raw, "trailing newline:", raw.endswith(b"\n"))
rec = json.loads(raw.decode("utf-8"))
A = "docs/project/maintainers/qualification/iiac_evidence/PDC-X/MCP-RC/activation/"
man = A + "iiac_audit_session_manifest.IIAC-ACT-N11-MCP-RC-ACTIVATION.json"
hof = A + "IIAC_SESSION_HANDOFF_RECORD_PDC_X_MCP_RC_ACTIVATION.md"
dec = "docs/project/maintainers/qualification/IIAC_OD_ACTIVATION_MCP_RC_2026-11-13.md"
line = [l for l in Path(dec).read_text(encoding="utf-8").splitlines() if l.startswith("Approved IIAC policy activation")]
assert len(line) == 1
rec.update({
    "effective_from_sha": "",
    "policy_activation_sha": "",
    "corrective_implementation_sha": "324d9dba37cb54e1b74a79a05148d2dcc05e6d09",
    "audit_execution_sha": "29d66ae3fcc7a06a919d7e07541740841e8abfe8",
    "evidence_publication_sha": "29d66ae3fcc7a06a919d7e07541740841e8abfe8",
    "certification_publication_sha": "7801dcb1293e31ee0c483184efd59c01a2dc606b",
    "session_b_verdict": "PASS",
    "session_b_manifest_digest": hashlib.sha256(Path(man).read_bytes()).hexdigest(),
    "session_b_manifest_path": man,
    "handoff_record_path": hof,
    "handoff_digest": hashlib.sha256(Path(hof).read_bytes()).hexdigest(),
    "operator_decision_path": dec,
    "operator_decision_id": "IIAC-OD-MCP-RC-ACT-N11-2026-11-13",
    "operator_activation_line": line[0],
    "operator_decision_class": "OD-MCP-RC-ACT-N11",
    "notes": "Phase 1 ACT-N11 MCP-RC integrated-baseline rebind (158 product paths versus the previous pin 2d3eceb6c, audited tree 324d9dba3). Previous pin (P6 activation, OD IIAC-OD-MCP-ACT-N11-P6-2026-11-11, phase-2 19c617365) preserved in history. The bound verdict PASS is the activation certifier's own, authored from PASS_SCOPED audits (scope limits in the binding manifest limitations and in the decision). effective_from_sha and policy_activation_sha are set in phase 2.",
})
out = json.dumps(rec, indent=2, ensure_ascii=False) + ("\n" if raw.endswith(b"\n") else "")
if b"\r\n" in raw:
    out = out.replace("\n", "\r\n")
R.write_bytes(out.encode("utf-8"))
