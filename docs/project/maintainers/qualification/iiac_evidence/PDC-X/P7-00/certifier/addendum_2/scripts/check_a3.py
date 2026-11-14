"""Certifier preflight addendum 2 (PDC-P7-00): git-derived checks for audit addendum 3.

Read-only. Run from the worktree root: python <this file>. Uses the activation gate's own
helpers (product_paths_changed_between, git_merge_base_is_ancestor, _material_changes).
"""
import hashlib, json, pathlib, subprocess, sys

sys.path.insert(0, ".")
from scripts.governance.iiac_common import product_paths_changed_between as f, git_merge_base_is_ancestor as anc
from scripts.governance.iiac_activation_gate import _material_changes


def sh(*a, raw=False):
    p = subprocess.run(a, capture_output=True)
    return p.stdout if raw else p.stdout.decode("utf-8", "replace")


H = sh("git", "rev-parse", "HEAD").strip()
PIN = "2d3eceb6cf7e93dccd4d96919319a6c969b4444b"
FA6 = "fa6ff1c9616a6df28c11b087eb3f1b0dc4097b5f"
A1 = "19d552beeb6f0677a8bdc3b4ce65da15dd637c7b"
A2 = "324d9dba37cb54e1b74a79a05148d2dcc05e6d09"
A3 = "0fc8badd81c4c3e46a3d0ce262c08e8efba41898"
EV2 = "29d66ae3fcc7a06a919d7e07541740841e8abfe8"
EV3 = "d5c51df0ef2ae63730129e673b426937867c1568"
CERT1 = "7801dcb1293e31ee0c483184efd59c01a2dc606b"
GOV = "2ae1014116d353bc7742985e62f2bccc2bfac798"
PHASE1 = "92d7308213bd46e76ce0b545cd94b36bc65d8e0e"
PHASE2 = "9949fe2c3dd64064f3b2002565b31c3c80b8cad6"
REG = "intergrax/integrations/providers/embedding_provider/ollama/register.py"
P = "docs/project/maintainers/qualification/iiac_evidence/PDC-X/P7-00/"
out = {"head": H}

s1, s2, s3, s4, sh_ = (set(f(PIN, x)) for x in (FA6, A1, A2, A3, H))
mc = set(f("6a6279a49a9aa5a33936a1c6a577c7194b4bcebb", "b0e19bf61b346e6941b2e190b703c6922e708b34"))
d_a2_a3 = f(A2, A3)
out["union"] = {
    "fa6ff1c96": len(s1), "19d552bee": len(s2), "324d9dba3": len(s3), "0fc8badd8": len(s4), "head": len(sh_),
    "all_equal_324_0fc_head": s3 == s4 == sh_,
    "mcp_rc": len(mc), "mcp_rc_subset": mc <= sh_, "non_mcp": len(sh_ - mc),
    "delta_324_0fc": d_a2_a3, "delta_path_already_in_union_at_324": REG in s3,
    "product_diff_324_to_head": f(A2, H), "product_diff_0fc_to_head": f(A3, H),
    "ancestors": {"324_anc_0fc": anc(A2, A3), "0fc_anc_head": anc(A3, H), "pin_anc_head": anc(PIN, H),
                  "ev3_anc_head": anc(EV3, H), "ev2_anc_0fc": anc(EV2, A3)},
    "mcp_rc_byte_identical_b0e_head": subprocess.run(["git", "diff", "--quiet", "b0e19bf61", H, "--"] + sorted(mc)).returncode == 0,
}
out["non_docs_changes_324_to_head"] = [l for l in sh("git", "diff", "--name-only", A2, H).splitlines() if l and not l.startswith("docs/")]
out["non_docs_non_ticket_changes_0fc_to_head"] = [l for l in sh("git", "diff", "--name-only", A3, H).splitlines() if l and not l.startswith("docs/")]
out["material_governance_changes_since"] = {x: _material_changes(pathlib.Path("."), x, H) for x in (PIN, A2, EV2, A3)}

# manifest digests
mpath = P + "addendum_3/iiac_audit_session_manifest.PDC-X.p7_00.addendum_3.json"
m = json.loads(pathlib.Path(mpath).read_text(encoding="utf-8"))
AUD = sh("git", "rev-parse", "origin/p7-00-baseline-audit").strip()
out["auditor_branch"] = AUD
h = lambda b: hashlib.sha256(b).hexdigest()
dig = {}
for dk, pk in {"audit_report_digest": "audit_report_path", "audit_scope_manifest_digest": "audit_scope_path",
               "test_matrix_digest": "test_matrix_path", "findings_digest": "findings_path"}.items():
    dig[dk] = {"claimed": m[dk],
               "at_head": h(pathlib.Path(m[pk]).read_bytes()) == m[dk],
               "at_evidence_commit_d5c51df0e": h(sh("git", "show", f"{EV3}:{m[pk]}", raw=True)) == m[dk],
               "on_auditor_branch": h(sh("git", "show", f"{AUD}:{m[pk]}", raw=True)) == m[dk]}
out["digests"] = dig
old = json.loads(sh("git", "show", f"{EV3}:{mpath}"))
aud = json.loads(sh("git", "show", f"{AUD}:{mpath}"))
out["pin_only_diff_evidence_to_head"] = {k: [old.get(k), m.get(k)] for k in sorted(set(old) | set(m)) if old.get(k) != m.get(k)}
out["pin_only_fields_vs_auditor_branch"] = {k: [aud.get(k), m.get(k)] for k in sorted(set(aud) | set(m)) if aud.get(k) != m.get(k)}
files = [l for l in sh("git", "diff", "--name-only", f"{EV3}~1", EV3).splitlines() if l]
diff = [x for x in files if sh("git", "rev-parse", f"{EV3}:{x}").strip() != sh("git", "rev-parse", f"{AUD}:{x}").strip()]
on_mcp_before = [x for x in files if sh("git", "cat-file", "-e", f"{A3}:{x}") == "" and subprocess.run(["git", "cat-file", "-e", f"{A3}:{x}"], capture_output=True).returncode == 0]
out["blob_identity"] = {"files_in_publication_commit": len(files), "differ_from_auditor_branch": diff,
                        "all_under_docs": all(x.startswith("docs/") for x in files), "already_present_on_0fc8badd8": on_mcp_before}
# auditor branch adds relative to 0fc8badd8 and relative to origin/mcp-rc
adds_a3 = {l for l in sh("git", "diff", "--name-only", "--diff-filter=A", A3, AUD).splitlines() if l}
adds_head = {l for l in sh("git", "diff", "--name-only", "--diff-filter=A", "origin/mcp-rc", AUD).splitlines() if l}
out["auditor_branch_adds"] = {"relative_to_0fc8badd8": len(adds_a3), "equals_import_set": adds_a3 == set(files),
                              "relative_to_origin_mcp_rc_head": len(adds_head)}
out["evidence_to_head_changes_outside_addendum3"] = [
    l for l in sh("git", "diff", "--name-only", EV3, H).splitlines()
    if l and not l.endswith("EVIDENCE_INTEGRATION_RECORD.md") and "addendum_3/iiac_audit_session_manifest" not in l]
out["earlier_manifests_unchanged_since_fde556183"] = sh(
    "git", "diff", "--name-only", "fde556183", H, "--", P + "iiac_audit_session_manifest.PDC-X.p7_00.json", P + "addendum_1", P + "addendum_2").strip() == ""
ids = {"mcp_rc_aud1": "MCP-RC-G6-AUDITOR-90803ebc-262a-457b-9315-14a5be0713d2",
       "mcp_rc_aud2": "MCP-RC-G6-REAUDIT-AUDITOR-9579becc-3291-43df-b623-f406ec80ac48",
       "p7_aud": "PDC-P7-00-ACT-N11-G6-BASELINE-AUDITOR-85fb061a-028f-4648-a3d6-284a392f4e27",
       "mcp_rc_impl": "T-IGX-038-A-F-MCP-RC-IMPLEMENTER"}
out["sessions"] = {"implementation": m["implementation_session_id"], "auditor": m["auditor_session_id"], "certifier": m["certifier_session_id"],
                   "all_distinct": len({m["implementation_session_id"], m["auditor_session_id"], m["certifier_session_id"]}) == 3,
                   "auditor_is_p7_auditor_plus_addendum_suffix": m["auditor_session_id"].startswith(ids["p7_aud"] + "-ADDENDUM-"),
                   "auditor_equals_mcp_rc_auditors": [k for k, v in ids.items() if v == m["auditor_session_id"]],
                   "impl_contains_mcp_rc_implementer": ids["mcp_rc_impl"] in m["implementation_session_id"],
                   "certifier_is_reserved_designation_not_session": m["certifier_session_id"] == "PDC-P7-00-CERTIFIER-PREFLIGHT-2026-11-13"}
out["manifest_fields"] = {k: m[k] for k in ("audited_implementation_sha", "test_execution_sha", "audit_execution_sha",
                                            "evidence_publication_sha", "certification_publication_sha", "verdict", "dossier_verdict")}

# re-pin simulation with the gate's own rules (what the gate would check for candidate pins)
def rep(corr, audx, evp, cert, effective=None):
    r = {"corr_anc_audx": anc(corr, audx), "audx_anc_evp": anc(audx, evp), "evp_anc_head": anc(evp, H), "cert_anc_head": anc(cert, H) if cert else None,
         "gov_anc_corr": anc(GOV, corr), "cert_ne_corr": cert != corr,
         "audx_to_head_product_paths": f(audx, H), "audx_to_head_material_paths": _material_changes(pathlib.Path("."), audx, H),
         "strict_product_diff_corr_vs_audx": f(corr, audx) if corr != audx else []}
    return r
out["repin_simulation_at_head"] = {
    "A_corrective_0fc8badd8_audit_exec_0fc8badd8_evidence_d5c51df0e": rep(A3, A3, EV3, CERT1),
    "B_corrective_0fc8badd8_audit_exec_d5c51df0e_evidence_d5c51df0e": rep(A3, EV3, EV3, CERT1),
    "C_corrective_324d9dba3_audit_exec_d5c51df0e_evidence_d5c51df0e_(wrong)": rep(A2, EV3, EV3, CERT1),
    "D_current_record_values_(corrective_324d9dba3_audit_exec_29d66ae3f)": rep(A2, EV2, EV2, CERT1),
}
print(json.dumps(out, indent=1))
json.dump(out, open(".tmp/session/p7-cert-a2/addendum3_checks.json", "w"), indent=1)
