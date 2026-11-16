"""Certifier preflight addendum 3 (PDC-P7-00): git-derived checks for audit addendum 4.

Read-only. Run from the worktree root. The auditor's evidence is read from the auditor branch
(origin/p7-00-baseline-audit) because it is not yet integrated into origin/mcp-rc.
"""
import hashlib, json, pathlib, re, subprocess, sys

sys.path.insert(0, ".")
from scripts.governance.iiac_common import product_paths_changed_between as f, git_merge_base_is_ancestor as anc
from scripts.governance.iiac_activation_gate import _material_changes, MATERIAL_GOVERNANCE_PATHS


def sh(*a, raw=False):
    p = subprocess.run(a, capture_output=True)
    return p.stdout if raw else p.stdout.decode("utf-8", "replace")


H = sh("git", "rev-parse", "HEAD").strip()
PIN = "2d3eceb6cf7e93dccd4d96919319a6c969b4444b"
A2 = "324d9dba37cb54e1b74a79a05148d2dcc05e6d09"
EV2 = "29d66ae3fcc7a06a919d7e07541740841e8abfe8"
A3 = "0fc8badd81c4c3e46a3d0ce262c08e8efba41898"
M4a = "0cb5dd011f363a7318246e907a1d307ad968db38"
A4 = "d02c399508c40b4a06d846410334d677f226c18d"
GOV = "2ae1014116d353bc7742985e62f2bccc2bfac798"
AUDB = sh("git", "rev-parse", "origin/p7-00-baseline-audit").strip()
Q = "docs/project/maintainers/qualification/"
P4 = Q + "iiac_evidence/PDC-X/P7-00/addendum_4/"
PROTO = Q + "INTEGRAX_INTERNAL_INDEPENDENT_AGENT_CERTIFICATION_PROTOCOL.md"
out = {"head": H, "auditor_branch": AUDB}

s158 = set(f(PIN, "0fc8badd81c4c3e46a3d0ce262c08e8efba41898"))
sH = set(f(PIN, H))
sA4 = set(f(PIN, A4))
mc = set(f("6a6279a49a9aa5a33936a1c6a577c7194b4bcebb", "b0e19bf61b346e6941b2e190b703c6922e708b34"))
scope = json.loads(sh("git", "show", f"{AUDB}:{P4}audit_scope.json"))
d_a3 = sorted(f(A3, H))
out["union"] = {
    "pin_to_0fc8badd8": len(s158), "pin_to_d02c39950": len(sA4), "pin_to_head": len(sH), "head_equals_d02c39950": sH == sA4,
    "new_vs_0fc8badd8": sorted(sH - s158), "removed_vs_0fc8badd8": sorted(s158 - sH),
    "mcp_rc": len(mc), "mcp_rc_subset": mc <= sH, "non_mcp": len(sH - mc),
    "delta_0fc8badd8_to_head": d_a3, "delta_equals_scope_claim": d_a3 == sorted(scope["delta_act_n11_paths_vs_0fc8badd8"]),
    "vs_324d9dba3": sorted(f(A2, H)), "vs_324d9dba3_equals_scope_claim": sorted(f(A2, H)) == sorted(scope["act_n11_paths_vs_324d9dba3"]),
    "vs_29d66ae3f_record_pin": sorted(f(EV2, H)),
    "ollama_register_unchanged_since_0fc8badd8": "intergrax/integrations/providers/embedding_provider/ollama/register.py" not in d_a3,
    "product_diff_0cb5dd011_to_head": f(M4a, H), "product_diff_d02c39950_to_head": f(A4, H),
    "mcp_rc_byte_identical_b0e_head": subprocess.run(["git", "diff", "--quiet", "b0e19bf61", H, "--"] + sorted(mc)).returncode == 0,
    "ancestors": {"0fc8badd8_anc_d02c39950": anc(A3, A4), "0cb5dd011_anc_d02c39950": anc(M4a, A4), "d02c39950_anc_head": anc(A4, H),
                  "auditor_head_contains_d02c39950": anc(A4, AUDB)},
}
out["non_docs_changes_0fc8badd8_to_head"] = [l for l in sh("git", "diff", "--name-only", A3, H).splitlines() if l and not l.startswith("docs/")]
out["non_docs_changes_0cb5dd011_to_head"] = [l for l in sh("git", "diff", "--name-only", M4a, H).splitlines() if l and not l.startswith("docs/")]
out["material_governance_changes"] = {x: _material_changes(pathlib.Path("."), x, H) for x in (EV2, A3, M4a, A4)}
out["protocol"] = {"numstat_0cb5dd011_to_head": sh("git", "diff", "--numstat", M4a, H, "--", PROTO).strip(),
                   "changed_lines_removed": [l for l in sh("git", "diff", "-U0", M4a, H, "--", PROTO).splitlines() if l.startswith("-") and not l.startswith("---")],
                   "changed_lines_added": [l for l in sh("git", "diff", "-U0", M4a, H, "--", PROTO).splitlines() if l.startswith("+") and not l.startswith("+++")],
                   "is_material_path": PROTO.replace(Q, Q) in MATERIAL_GOVERNANCE_PATHS,
                   "kept_sentences_present": {s: s in pathlib.Path(PROTO).read_text(encoding="utf-8") for s in (
                       "`IN_FORCE` requires the fail-closed gate (`--require-in-force`) and a separate operator activation record.",
                       "a certification session after the ACT-01 gate correction is required.", "All required before `IN_FORCE`:")}}
out["gate_scripts_changed_since_pin"] = [l for l in sh("git", "diff", "--name-only", PIN, H, "--", "scripts/governance", ".github").splitlines() if l]
out["gate_scripts_changed_0fc8badd8_to_head"] = [l for l in sh("git", "diff", "--name-only", A3, H, "--", "scripts/governance", "scripts/maintenance", "scripts/enterprise_audit", "agent-env", Q + "IIAC_POLICY_ACTIVATION_RECORD.json", "pyproject.toml", "uv.lock").splitlines() if l]

# tests: weakened assertions in test diff 0fc8badd8..HEAD
td = sh("git", "diff", "-U0", A3, H, "--", "tests", "applications/local_workspace_application/tests")
cur = None; removed_assert = []; added_skip = []
for l in td.splitlines():
    if l.startswith("+++ b/"): cur = l[6:]
    elif l.startswith("-") and not l.startswith("---") and re.search(r"\bassert\b|pytest\.raises|xfail|skipif|\.skip\(", l): removed_assert.append(f"{cur}: {l[:160]}")
    elif l.startswith("+") and not l.startswith("+++") and re.search(r"xfail|pytest\.skip|mark\.skip|skipif|importorskip", l): added_skip.append(f"{cur}: {l[:160]}")
out["test_diff_scan"] = {"files_changed": len([l for l in sh('git', 'diff', '--name-only', A3, H, '--', 'tests', 'applications/local_workspace_application/tests').splitlines() if l]),
                         "removed_assert_or_xfail_lines": removed_assert, "added_skip_xfail_importorskip_lines": added_skip}
out["f41_test_files_unchanged_0fc8badd8_to_head"] = sh("git", "diff", "--name-only", A3, H, "--",
    "tests/qualification/pdc_x/xd/test_xd_12_evol_ent_audit.py", "tests/qualification/pdc_x/p6_impact/test_p6_impact_map_coverage.py").strip() == ""

# auditor evidence (on the auditor branch)
mpath = P4 + "iiac_audit_session_manifest.PDC-X.p7_00.addendum_4.json"
m = json.loads(sh("git", "show", f"{AUDB}:{mpath}"))
h = lambda b: hashlib.sha256(b).hexdigest()
dig = {}
for dk, pk in {"audit_report_digest": "audit_report_path", "audit_scope_manifest_digest": "audit_scope_path", "test_matrix_digest": "test_matrix_path", "findings_digest": "findings_path"}.items():
    dig[dk] = {"claimed": m[dk], "on_auditor_branch": h(sh("git", "show", f"{AUDB}:{m[pk]}", raw=True)) == m[dk]}
out["digests_on_auditor_branch"] = dig
out["manifest_fields"] = {k: m[k] for k in ("audited_implementation_sha", "test_execution_sha", "audit_execution_sha", "evidence_publication_sha", "certification_publication_sha", "certifier_session_id", "verdict", "dossier_verdict")}
ids = {"mcp_rc_aud1": "MCP-RC-G6-AUDITOR-90803ebc-262a-457b-9315-14a5be0713d2", "mcp_rc_aud2": "MCP-RC-G6-REAUDIT-AUDITOR-9579becc-3291-43df-b623-f406ec80ac48",
       "p7_aud": "PDC-P7-00-ACT-N11-G6-BASELINE-AUDITOR-85fb061a-028f-4648-a3d6-284a392f4e27", "mcp_rc_impl": "T-IGX-038-A-F-MCP-RC-IMPLEMENTER"}
out["sessions"] = {"implementation": m["implementation_session_id"], "auditor": m["auditor_session_id"],
                   "auditor_is_p7_auditor_plus_suffix": m["auditor_session_id"].startswith(ids["p7_aud"] + "-ADDENDUM-"),
                   "auditor_equals_mcp_rc_auditors": [k for k, v in ids.items() if v == m["auditor_session_id"]],
                   "impl_contains_mcp_rc_implementer": ids["mcp_rc_impl"] in m["implementation_session_id"],
                   "impl_label_equals_addendum_2_and_3": m["implementation_session_id"] == "T-IGX-036-J-V-PDC-X-IMPLEMENTERS+T-IGX-038-A-F-MCP-RC-IMPLEMENTER"}
fl = json.loads(sh("git", "show", f"{AUDB}:{m['findings_path']}"))
out["findings"] = {"ids": [x["id"] for x in fl["findings"]], "counts_claimed": fl["counts_new"],
                   "counts_recomputed": {s: sum(1 for x in fl["findings"] if x["severity"] == s) for s in ("HIGH", "MEDIUM", "LOW", "INFO")},
                   "blocking_open": [x["id"] for x in fl["findings"] if x["severity"] in ("BLOCKER", "CRITICAL")]}
# evidence on mcp-rc?
out["addendum_4_evidence_on_mcp_rc_head"] = sh("git", "ls-tree", "-r", "--name-only", H, "--", P4).strip() != ""
adds = {l for l in sh("git", "diff", "--name-only", "--diff-filter=A", H, AUDB).splitlines() if l}
out["auditor_branch_adds_vs_head"] = {"count": len(adds), "all_under_docs": all(x.startswith("docs/") for x in adds), "outside_addendum_4_and_dossier": sorted(x for x in adds if "/addendum_4/" not in x and not x.endswith("BASELINE_AUDIT_ADDENDUM_4.md"))}

# re-pin simulation
def rep(corr, audx, evp, cert):
    return {"corr_anc_audx": anc(corr, audx), "audx_anc_evp": anc(audx, evp), "evp_anc_head": anc(evp, H), "cert_anc_head": anc(cert, H), "gov_anc_corr": anc(GOV, corr),
            "audx_to_head_product_paths": f(audx, H), "audx_to_head_material_paths": _material_changes(pathlib.Path("."), audx, H),
            "strict_product_diff_corr_vs_audx": f(corr, audx) if corr != audx else []}
CERT2 = "0c79c9f6519473eaf3c5a9e1fbf678378d6f1690"
out["repin_simulation_at_head_evidence_placeholder_HEAD"] = {
    "E_corr_d02c39950_audx_d02c39950": rep(A4, A4, H, CERT2),
    "F_corr_d02c39950_audx_HEAD_evidence_HEAD": rep(A4, H, H, CERT2),
    "G_corr_0fc8badd8_audx_d02c39950_(strict product diff)": rep(A3, A4, H, CERT2),
    "H_corr_d02c39950_audx_0cb5dd011_(before the protocol edit)": rep(A4, M4a, H, CERT2),
    "I_current_record_values": rep(A2, EV2, EV2, "7801dcb1293e31ee0c483184efd59c01a2dc606b"),
}
print(json.dumps(out, indent=1))
