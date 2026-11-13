import json, re
P="docs/project/maintainers/qualification/iiac_evidence/PDC-X/P7-00/"
a=json.load(open(P+"iiac_audit_p7_00_findings.json",encoding="utf-8"))
b=json.load(open(P+"addendum_1/iiac_audit_p7_00_addendum_1_findings.json",encoding="utf-8"))
allf=a["findings"]+b["findings"]
ids=[x["id"] for x in allf]
exp=[f"F-{i:02d}" for i in range(1,41)]
res={"ids_complete":ids==exp,"missing":sorted(set(exp)-set(ids)),"dupes":sorted({i for i in ids if ids.count(i)>1}),
"no_disposition":[x["id"] for x in allf if not (x.get("recommended_disposition") or x.get("disposition"))],
"blocking_open":[x["id"] for x in allf if str(x.get("severity")).upper() in ("BLOCKER","CRITICAL")],
"severity_counts":{s:sum(1 for x in allf if x["severity"]==s) for s in ("HIGH","MEDIUM","LOW","INFO")},
"first_counts_claim":a["counts"],"addendum_counts_claim":b["counts_new"]}
routes={}
for k,v in a["routed_to_domain_audits"].items():
    for i in v: routes.setdefault(i,[]).append(k)
rows=[]
for x in allf:
    d=(x.get("recommended_disposition") or x.get("disposition") or "")
    cat=("ACCEPT" if d.upper().startswith("ACCEPT") else "ROUTE" if d.upper().startswith("ROUTE") else "TEST_ONLY" if "TEST_ONLY" in d.upper() else "RECOMMEND" if d.upper().startswith("RECOMMEND") else "OTHER")
    rows.append({"id":x["id"],"sev":x["severity"],"area":x.get("area"),"title":x["title"],"disposition":d,"category":cat,"routed_first_audit":routes.get(x["id"],[]),"verification":x.get("verification")})
res["rows"]=rows
res["category_counts"]={c:sum(1 for r in rows if r["category"]==c) for c in set(r["category"] for r in rows)}
# routed_to_domain_audits ids all exist
res["routed_ids_exist"]=all(i in ids for v in a["routed_to_domain_audits"].values() for i in v)
res["addendum_status"]=b["status_of_first_audit_findings"]
json.dump(res,open(".tmp/session/p7-00-cert/findings_disposition.json","w"),indent=1)
print({k:v for k,v in res.items() if k not in ("rows","addendum_status")})
for r in rows:
    if r["category"] in ("TEST_ONLY","RECOMMEND","OTHER") or r["id"] in ("F-04","F-05","F-07","F-09"): print(r["id"],r["sev"],r["category"],"|",r["disposition"][:230])
