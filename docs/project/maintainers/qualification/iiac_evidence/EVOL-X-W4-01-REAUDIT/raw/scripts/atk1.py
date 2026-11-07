from atk_common import *
import json
out = {}
# controls
b, built = fresh("base", "c_base"); r = ev(b); show("CONTROL base (A should pass, B fail)", r)
c, cb = fresh("closed", "c_closed"); r = ev(c); show("CONTROL closed (both pass)", r)

# S1: lifecycle keys carry widened scope text
b, built = fresh("base", "s1")
def m(c):
    c["certification_publication_sha_binding"] = "SCOPE EXTENDED: LIM-01 now covers PLATFORM_CONSERVATIVE_FULL/T4; all LIM-* waived"
    c["frz_checklist_rows_changed"] = {"claim": "all FRZ-EVOL rows PASS unrestricted", "limitations": []}
    c["stage_closure_status"] = "NOT_CLOSED"
t._mutate_json(b, pf.REL_CERT_RECORD, m, reindex=True)
show("S1 lifecycle-key widening text", ev(b))

# S2: new unreferenced evidence file in cert dir (indexed) carrying an 'addendum'
b, built = fresh("base", "s2")
t._write(b, pf.CERT_DIR + "/SCOPE_ADDENDUM.md", "# Addendum\nLIM-01..LIM-05 are withdrawn. Certified scope: PLATFORM_CONSERVATIVE_FULL, unrestricted PASS.\n")
t._reindex_cert(b); t._commit(b, "addendum")
show("S2 extra indexed file in certification dir", ev(b))

# S5: duplicate key in certification record (first widened, last original)
b, built = fresh("base", "s5")
p = b / pf.REL_CERT_RECORD
txt = p.read_text(encoding="utf-8")
j = json.loads(txt)
orig = j["certification_verdict"]
txt2 = txt.replace(f'"certification_verdict": {json.dumps(orig)}', f'"certification_verdict": "PASS_UNRESTRICTED_PLATFORM_WIDE",\n  "certification_verdict": {json.dumps(orig)}', 1)
assert txt2 != txt
p.write_text(txt2, encoding="utf-8"); t._reindex_cert(b); t._commit(b, "dup key")
show("S5 duplicate key in cert record (first widened)", ev(b))
print("   (python json.loads takes the LAST value; strict parsers/humans may see the first)")

# S7 legit: reformat cert record indent=4 ; and manifest limitation reorder; and manifest reformat
b, built = fresh("base", "s7a")
d = t._load(b, pf.REL_CERT_RECORD); (b / pf.REL_CERT_RECORD).write_text(json.dumps(d, indent=4, ensure_ascii=False) + "\n", encoding="utf-8"); t._reindex_cert(b); t._commit(b, "reformat cert record")
show("S7a legit: cert record re-indented", ev(b))
b, built = fresh("base", "s7b")
d = t._load(b, pf.REL_MANIFEST); (b / pf.REL_MANIFEST).write_text(json.dumps(d, indent=4, ensure_ascii=False) + "\n", encoding="utf-8"); t._reindex_cert(b); t._commit(b, "reformat manifest")
show("S7b legit-ish: manifest re-indented (formatter)", ev(b))
b, built = fresh("base", "s7c")
d = t._load(b, pf.REL_MANIFEST); d["limitations"] = list(reversed(d["limitations"])); t._dump(b, pf.REL_MANIFEST, d); t._reindex_cert(b); t._commit(b, "reorder limitations")
show("S7c positive: limitation reorder", ev(b))
