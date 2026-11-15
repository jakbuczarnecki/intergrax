import hashlib, json, pathlib
W = pathlib.Path(".tmp/session/mcp-rc-cert/auditfiles")
out = {}
for tag in ("a1","a2"):
    base = W/tag
    man = next(base.rglob("iiac_audit_session_manifest*.json"))
    m = json.loads(man.read_text(encoding="utf-8"))
    fields = {"audit_report_digest":"audit_report_path","audit_scope_manifest_digest":"audit_scope_path","test_matrix_digest":"test_matrix_path","findings_digest":"findings_path"}
    res = {}
    for dk, pk in fields.items():
        p = base/m[pk]
        raw = p.read_bytes()
        cands = {"raw": hashlib.sha256(raw).hexdigest(),
                 "lf": hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest(),
                 "plan_normalized": None}
        import sys; sys.path.insert(0, ".")
        from scripts.governance.plan_digest import digest_text
        cands["plan_normalized"] = digest_text(raw.decode("utf-8"))
        match = [k for k,v in cands.items() if v == m[dk]]
        res[dk] = {"claimed": m[dk], "file": m[pk], "match": match or "NO_MATCH", "computed": cands}
    out[tag] = res
    print(tag, {k: v["match"] for k,v in res.items()})
pathlib.Path(".tmp/session/mcp-rc-cert/digest_check.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
