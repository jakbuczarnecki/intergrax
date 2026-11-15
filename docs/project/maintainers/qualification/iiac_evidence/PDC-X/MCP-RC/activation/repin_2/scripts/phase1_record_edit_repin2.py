import json, hashlib
from pathlib import Path
R = Path("docs/project/maintainers/qualification/IIAC_POLICY_ACTIVATION_RECORD.json")
raw = R.read_bytes()
rec = json.loads(raw.decode("utf-8"))
A = "docs/project/maintainers/qualification/iiac_evidence/PDC-X/MCP-RC/activation/repin_2/"
man = A + "iiac_audit_session_manifest.IIAC-ACT-N11-MCP-RC-ACTIVATION-REPIN-2.json"
hof = A + "IIAC_SESSION_HANDOFF_RECORD_PDC_X_MCP_RC_ACTIVATION_REPIN_2.md"
rec.update({
    "effective_from_sha": "",
    "policy_activation_sha": "",
    "corrective_implementation_sha": "d02c399508c40b4a06d846410334d677f226c18d",
    "audit_execution_sha": "a1b3053b49c28124a0abde679d733f21a597797f",
    "evidence_publication_sha": "a1b3053b49c28124a0abde679d733f21a597797f",
    "certification_publication_sha": "dc1e66fa140d81b28c2658e88b4e1aa16eb29ad2",
    "session_b_verdict": "PASS",
    "session_b_manifest_digest": hashlib.sha256(Path(man).read_bytes()).hexdigest(),
    "session_b_manifest_path": man,
    "handoff_record_path": hof,
    "handoff_digest": hashlib.sha256(Path(hof).read_bytes()).hexdigest(),
    "notes": "Phase 1 ACT-N11 MCP-RC re-pin 2 for the final integrated tree (161 product paths versus the original pin 2d3eceb6c, audited tree d02c39950, development 83d939095 merged). Previous bindings preserved in history: P6 activation (phase-2 19c617365) and the first MCP-RC binding (phase-2 9949fe2c3). The signed operator line is reused unchanged (OD-MCP-RC-05). The bound verdict PASS is the activation certifier's own, authored from PASS_SCOPED audits (scope limits in the binding manifest limitations and in the decision). effective_from_sha and policy_activation_sha are set in phase 2.",
})
out = json.dumps(rec, indent=2, ensure_ascii=False) + "\n"
R.write_bytes(out.encode("utf-8"))
