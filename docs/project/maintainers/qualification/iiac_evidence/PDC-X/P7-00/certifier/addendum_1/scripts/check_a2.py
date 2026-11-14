import hashlib, json, subprocess, sys, pathlib
sys.path.insert(0, ".")
from scripts.governance.iiac_common import product_paths_changed_between as f, git_merge_base_is_ancestor as anc
def sh(*a, raw=False):
    p = subprocess.run(a, capture_output=True); return p.stdout if raw else p.stdout.decode("utf-8","replace")
H = sh("git","rev-parse","HEAD").strip()
PIN="2d3eceb6cf7e93dccd4d96919319a6c969b4444b"; A1="19d552beeb6f0677a8bdc3b4ce65da15dd637c7b"; A2="324d9dba37cb54e1b74a79a05148d2dcc05e6d09"
P = "docs/project/maintainers/qualification/iiac_evidence/PDC-X/P7-00/"
out = {"head": H}
s1=set(f(PIN,"fa6ff1c9616a6df28c11b087eb3f1b0dc4097b5f")); s2=set(f(PIN,A1)); s3=set(f(PIN,A2)); sh_=set(f(PIN,H))
d=set(f(A1,A2)); mc=set(f("6a6279a49a9aa5a33936a1c6a577c7194b4bcebb","b0e19bf61b346e6941b2e190b703c6922e708b34"))
sc=json.load(open(P+"addendum_2/audit_scope.json",encoding="utf-8"))
out["union"]={"fa6":len(s1),"19d":len(s2),"324d9dba3":len(s3),"head":len(sh_),"head_equals_324":sh_==s3,"delta_19d_324":sorted(d),"delta_new":sorted(d-s2),
 "mcp_rc":len(mc),"mcp_rc_subset":mc<=s3,"non_mcp":len(s3-mc),
 "product_diff_324_vs_head":len(f(A2,H)),"product_diff_19d_vs_324":len(d),
 "ancestors":{"324_anc_head":anc(A2,H),"19d_anc_324":anc(A1,A2),"pin_anc_head":anc(PIN,H),"a156cbbd5_anc_324":anc("a156cbbd5ff521dc7e99dd02ab6bebd4ca234a73",A2),"884c177ad_anc_head":anc("884c177adc366004929641d029bba8d129b0ecc8",H)},
 "mcp_rc_byte_identical_b0e_head": subprocess.run(["git","diff","--quiet","b0e19bf61",H,"--"]+sorted(mc)).returncode==0}
out["scope_keys"]={k:(v if not isinstance(v,(list,dict)) else f"{type(v).__name__}({len(v)})") for k,v in sc.items()}
# check scope claims
for k in ("act_n11_paths","delta_act_n11_paths","combined_act_n11_paths"):
    if k in sc: out["union"]["scope_"+k+"_equals_git"]= (sorted(sc[k])==sorted(s3) if "combined" in k or k=="act_n11_paths" else sorted(sc[k])==sorted(d))
mpath = P+"addendum_2/iiac_audit_session_manifest.PDC-X.p7_00.addendum_2.json"
m = json.loads(pathlib.Path(mpath).read_text(encoding="utf-8"))
h=lambda b: hashlib.sha256(b).hexdigest()
EV="29d66ae3fcc7a06a919d7e07541740841e8abfe8"
AUD=sh("git","rev-parse","origin/p7-00-baseline-audit").strip()
out["auditor_branch"]=AUD
dig={}
for dk,pk in {"audit_report_digest":"audit_report_path","audit_scope_manifest_digest":"audit_scope_path","test_matrix_digest":"test_matrix_path","findings_digest":"findings_path"}.items():
    dig[dk]={"claimed":m[dk],"at_head":h(pathlib.Path(m[pk]).read_bytes())==m[dk],"at_evidence_commit":h(sh("git","show",f"{EV}:{m[pk]}",raw=True))==m[dk],"on_auditor_branch":h(sh("git","show",f"{AUD}:{m[pk]}",raw=True))==m[dk]}
out["digests"]=dig
old=json.loads(sh("git","show",f"{EV}:{mpath}")); aud=json.loads(sh("git","show",f"{AUD}:{mpath}"))
out["pin_only_diff_evidence_to_head"]={k:[old.get(k),m.get(k)] for k in sorted(set(old)|set(m)) if old.get(k)!=m.get(k)}
out["pin_only_fields_vs_auditor_branch"]=sorted(k for k in set(aud)|set(m) if aud.get(k)!=m.get(k))
files=[l for l in sh("git","diff","--name-only",f"{EV}~1",EV).splitlines() if l]
diff=[x for x in files if sh("git","rev-parse",f"{EV}:{x}").strip()!=sh("git","rev-parse",f"{AUD}:{x}").strip()]
out["blob_identity"]={"files_in_publication_commit":len(files),"differ_from_auditor_branch":diff,"all_under_docs":all(x.startswith("docs/") for x in files)}
out["changed_evidence_to_head"]=[l for l in sh("git","diff","--name-only",EV,H).splitlines() if l and not l.startswith("docs/")]
out["changed_evidence_to_head_docs_only_list"]=[l for l in sh("git","diff","--name-only",EV,H).splitlines() if l]
# earlier manifests untouched since 5c25d4d8f?
out["earlier_manifests_unchanged_since_5c25d4d8f"]=sh("git","diff","--name-only","5c25d4d8f",H,"--",P+"iiac_audit_session_manifest.PDC-X.p7_00.json",P+"addendum_1").strip()=="" 
# sessions
ids={"mcp_rc_aud1":"MCP-RC-G6-AUDITOR-90803ebc-262a-457b-9315-14a5be0713d2","mcp_rc_aud2":"MCP-RC-G6-REAUDIT-AUDITOR-9579becc-3291-43df-b623-f406ec80ac48","p7_aud1":"PDC-P7-00-ACT-N11-G6-BASELINE-AUDITOR-85fb061a-028f-4648-a3d6-284a392f4e27","mcp_rc_impl":"T-IGX-038-A-F-MCP-RC-IMPLEMENTER"}
out["sessions"]={"implementation":m["implementation_session_id"],"auditor":m["auditor_session_id"],"certifier":m["certifier_session_id"],
 "auditor_is_p7_auditor_plus_suffix":m["auditor_session_id"].startswith(ids["p7_aud1"]+"-ADDENDUM-"),"auditor_equals_mcp_rc_auditors":[k for k,v in ids.items() if v==m["auditor_session_id"]],
 "impl_contains_mcp_rc_implementer":ids["mcp_rc_impl"] in m["implementation_session_id"],"all_distinct":len({m["implementation_session_id"],m["auditor_session_id"],m["certifier_session_id"]})==3}
print(json.dumps(out,indent=1)); json.dump(out,open(".tmp/session/p7-00-cert/addendum2_checks.json","w"),indent=1)
