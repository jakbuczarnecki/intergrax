"""Read-only dry run of the manifest-side conditions of the activation gate (ACT-N03/N05/N08/N09/N10/N17), no record edit.

Run from the checkout root: python check_binding_manifest.py
Uses the gate's own schema validator and ancestry helper; the activation record is only READ for the bound digests.
"""
import hashlib
import json
import sys
from pathlib import Path

REPO = Path.cwd()
sys.path.insert(0, str(REPO))
from scripts.governance import iiac_activation_gate as G_
from scripts.governance.iiac_common import git_merge_base_is_ancestor

A = "docs/project/maintainers/qualification/iiac_evidence/PDC-X/P8/activation/"
M = Path(A + "iiac_audit_session_manifest.IIAC-ACT-N11-PDC-X-P8-04.json")
H = Path(A + "IIAC_SESSION_HANDOFF_RECORD_PDC_X_P8_04_ACTIVATION.md")
REC = json.loads(Path("docs/project/maintainers/qualification/IIAC_POLICY_ACTIVATION_RECORD.json").read_text(encoding="utf-8"))
m = json.loads(M.read_text(encoding="utf-8"))
ok_all = True


def ok(name, cond, detail=""):
    global ok_all
    print(f"[{'OK  ' if cond else 'FAIL'}] {name} {detail}")
    ok_all &= bool(cond)


errs = []
G_.validate_manifest_dict(m, errs)
ok("gate schema validation (validate_manifest_dict)", errs == [], str(errs))
ok("stage_id IIAC-GOV-1, policy IIAC-2026-11-05.1", m["stage_id"] == "IIAC-GOV-1" and m["policy_version"] == "IIAC-2026-11-05.1")
ok("verdict PASS (ACT-N05)", m["verdict"] == "PASS")
ok("audited_implementation_sha == test_execution_sha == G", m["audited_implementation_sha"] == m["test_execution_sha"] == "969cbd964c679d563b02e55f065b452d46fdf90b")
ok("audit_execution_sha == evidence_publication_sha == E", m["audit_execution_sha"] == m["evidence_publication_sha"] == "a73d15ab341b30e9748aad03894f6faa75ad4e09")
ok("certification_publication_sha != G", m["certification_publication_sha"] != m["audited_implementation_sha"])
ok("implementation id != auditor id, agent ids differ (ACT-N10)", m["implementation_session_id"] != m["auditor_session_id"] and m["implementation_agent_id"] != m["auditor_agent_id"])
ok("certifier and auditor ids non-empty (ACT-N17)", bool(m["certifier_session_id"].strip()) and bool(m["auditor_session_id"].strip()))
ok("certifier id differs from implementer and auditor", m["certifier_session_id"] not in (m["implementation_session_id"], m["auditor_session_id"]))
for field, key in (
    ("audit_report_digest", "audit_report_path"),
    ("audit_scope_manifest_digest", "audit_scope_path"),
    ("test_matrix_digest", "test_matrix_path"),
    ("findings_digest", "findings_path"),
):
    ok(f"{field} == sha256(bytes of record {key})", hashlib.sha256(Path(REC[key]).read_bytes()).hexdigest() == m[field])
ok("manifest digests == record bound digests", m["audit_report_digest"] == REC["session_b_governance_audit_report_digest"])
text = H.read_bytes().decode("utf-8")
ok("handoff contains policy version", "IIAC-2026-11-05.1" in text)
ok("handoff contains implementation session id verbatim (ACT-N17)", m["implementation_session_id"] in text)
ok("handoff contains auditor session id verbatim (ACT-N17)", m["auditor_session_id"] in text)
ok("handoff has CRLF line endings", b"\r\n" in H.read_bytes() and b"\n" not in H.read_bytes().replace(b"\r\n", b""))
for a, b in (("2ae1014116d353bc7742985e62f2bccc2bfac798", m["audited_implementation_sha"]), (m["audited_implementation_sha"], m["audit_execution_sha"]), (m["audit_execution_sha"], "HEAD"), (m["certification_publication_sha"], "HEAD"), (m["audit_execution_sha"], m["certification_publication_sha"])):
    ok(f"ancestor {a[:9]} -> {b[:9]}", git_merge_base_is_ancestor(a, b, repo=REPO))
print("limitations:", len(m["limitations"]), "manifest sha256:", hashlib.sha256(M.read_bytes()).hexdigest())
print("handoff sha256:", hashlib.sha256(H.read_bytes()).hexdigest())
print("ALL_MANIFEST_CHECKS_OK =", ok_all)
sys.exit(0 if ok_all else 1)
