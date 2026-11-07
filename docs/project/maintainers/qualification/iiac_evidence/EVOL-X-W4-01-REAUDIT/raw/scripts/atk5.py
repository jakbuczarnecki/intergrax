from atk_common import *
git = t._git
def run(label, b, p2):
    r = pf.evaluate(b, evidence_root=b, p2_sha=p2); 
    print(f"== {label}: gateA={'PASS' if r.gate_a_pass else 'FAIL'}"); [print("   A:", e[:200]) for e in r.gate_a[:3]]
# V1 own repoint: widen LIM-02 + verdict-in-record, rebind both fields to HEAD
b,_ = fresh("base","v1"); p2 = t._honest_p2(b)
def w(m): i=t._find_limitation(m,"LIM-02"); m["limitations"][i]+="; M-09H was measured and passed"
t._mutate_json(b, pf.REL_MANIFEST, w, reindex=True)
head = git(b,"rev-parse","HEAD"); t._rebind_certification_publication(b, head)
run("V1 widen LIM-02 + repoint manifest&record certification_publication_sha to widened commit", b, p2)
# V2 edit P2-listed assessment doc and regenerate index
b,_ = fresh("base","v2"); p2 = t._honest_p2(b)
names = [p.name for p in (b/pf.CERT_DIR).iterdir() if p.is_file() and p.suffix==".md"]
print("   cert dir md files:", names)
tgt = b/pf.CERT_DIR/names[0]; tgt.write_text(tgt.read_text(encoding="utf-8")+"\nAmendment: LIM-01 withdrawn.\n", encoding="utf-8")
t._reindex_cert(b); t._commit(b,"amend assessment")
run("V2 amend a P2 certification evidence file + reindex", b, p2)
# V3 flip SCOPED->UNSCOPED in disposition + repoint
b,_ = fresh("base","v3"); p2 = t._honest_p2(b)
t._mutate_json(b, pf.REL_CERT_RECORD, lambda c: c["frz_evol_dispositions"].__setitem__("FRZ-EVOL-09","ELIGIBLE_PASS"), reindex=True)
t._rebind_certification_publication(b, git(b,"rev-parse","HEAD"))
run("V3 FRZ-EVOL-09 SCOPED->UNSCOPED + repoint", b, p2)
# V4 delete the record from P2 history by creating new root-like orphan history? -> check first!=anchor
b,_ = fresh("base","v4"); p2 = t._honest_p2(b)
run("V4 control: honest base", b, p2)
