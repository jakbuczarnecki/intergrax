import hashlib, json, subprocess, sys, pathlib
sys.path.insert(0, ".")
def sh(*a, raw=False):
    p = subprocess.run(a, capture_output=True); 
    return p.stdout if raw else p.stdout.decode("utf-8","replace")
Q = "docs/project/maintainers/qualification/"
mans = {"first": Q+"iiac_evidence/PDC-X/P7-00/iiac_audit_session_manifest.PDC-X.p7_00.json",
        "addendum_1": Q+"iiac_evidence/PDC-X/P7-00/addendum_1/iiac_audit_session_manifest.PDC-X.p7_00.addendum_1.json"}
out = {"digests": {}, "pin_only_edits": {}, "blob_identity": {}, "sessions": {}}
M = {}
for tag, path in mans.items():
    m = json.loads(pathlib.Path(path).read_text(encoding="utf-8")); M[tag] = m
    for dk, pk in {"audit_report_digest":"audit_report_path","audit_scope_manifest_digest":"audit_scope_path","test_matrix_digest":"test_matrix_path","findings_digest":"findings_path"}.items():
        raw_head = pathlib.Path(m[pk]).read_bytes()
        raw_pub = sh("git","show",f"6b195980d:{m[pk]}", raw=True)
        raw_aud = sh("git","show",f"origin/p7-00-baseline-audit:{m[pk]}", raw=True)
        h = lambda b: hashlib.sha256(b).hexdigest()
        out["digests"][f"{tag}.{dk}"] = {"claimed": m[dk], "file_at_head": h(raw_head)==m[dk], "file_at_6b195980d": h(raw_pub)==m[dk], "file_on_auditor_branch": h(raw_aud)==m[dk]}
    # pin-only edit diff: manifest at 6b195980d vs head
    old = json.loads(sh("git","show",f"6b195980d:{path}")); new = m
    out["pin_only_edits"][tag] = {k: [old.get(k), new.get(k)] for k in sorted(set(old)|set(new)) if old.get(k) != new.get(k)}
    # vs auditor branch
    aud = json.loads(sh("git","show",f"origin/p7-00-baseline-audit:{path}"))
    out["pin_only_edits"][tag+"_vs_auditor_branch"] = sorted(k for k in set(aud)|set(new) if aud.get(k) != new.get(k))
# blob identity of the 56 imported files between 6b195980d and auditor branch
files = [l for l in sh("git","diff","--name-only","6b195980d~1","6b195980d").splitlines() if l]
same = 0; diffs = []
for f in files:
    a = sh("git","rev-parse",f"6b195980d:{f}").strip(); b = sh("git","rev-parse",f"origin/p7-00-baseline-audit:{f}").strip()
    (same:=same+1) if a == b else diffs.append(f)
out["blob_identity"] = {"files_in_publication_commit": len(files), "identical_to_auditor_branch": same, "differ": diffs}
# files at head vs at 6b195980d other than the 2 manifests + record
chg = [l for l in sh("git","diff","--name-only","6b195980d","ae5623efb").splitlines() if l]
out["changed_6b195980d_to_head"] = chg
# sessions
ids = {"implementer_mcp_rc":"T-IGX-038-A-F-MCP-RC-IMPLEMENTER","mcp_rc_auditor_1":"MCP-RC-G6-AUDITOR-90803ebc-262a-457b-9315-14a5be0713d2","mcp_rc_auditor_2":"MCP-RC-G6-REAUDIT-AUDITOR-9579becc-3291-43df-b623-f406ec80ac48","my_mcp_rc_preflight":"MCP-RC-CERTIFIER-PREFLIGHT-2026-11-13"}
for tag, m in M.items():
    out["sessions"][tag] = {"implementation": m["implementation_session_id"], "auditor": m["auditor_session_id"], "certifier": m["certifier_session_id"],
        "auditor_equals_any_known": [k for k,v in ids.items() if v == m["auditor_session_id"]],
        "implementation_contains_mcp_rc_implementer": ids["implementer_mcp_rc"] in m["implementation_session_id"],
        "auditor_differs_from_certifier": m["auditor_session_id"] != m["certifier_session_id"], "implementation_differs_from_auditor": m["implementation_session_id"] != m["auditor_session_id"]}
out["sessions"]["first_vs_addendum_auditor_distinct_strings"] = M["first"]["auditor_session_id"] != M["addendum_1"]["auditor_session_id"]
out["sessions"]["addendum_auditor_id_derived_from_first"] = M["addendum_1"]["auditor_session_id"].startswith(M["first"]["auditor_session_id"])
print(json.dumps(out, indent=1))
json.dump(out, open(".tmp/session/p7-00-cert/p7_manifest_checks.json","w"), indent=1)
