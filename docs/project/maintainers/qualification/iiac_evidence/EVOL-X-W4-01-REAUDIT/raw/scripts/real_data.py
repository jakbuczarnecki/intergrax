import sys, json, subprocess, os
sys.path.insert(0, r"D:/Projekty/intergrax/.claude/worktrees/stoic-panini-83e1a9")
os.chdir(r"D:/Projekty/intergrax/.claude/worktrees/stoic-panini-83e1a9")
from scripts.qualification import evol_x_closure_publication_preflight as pf
P2 = pf.P2_CERTIFICATION_PUBLICATION_SHA
raw = subprocess.run(["git","show",f"{P2}:{pf.REL_CERT_RECORD}"],capture_output=True).stdout
base = json.loads(raw)
print("P2 dispositions -> rendered promoted rows:")
for k, v in base["frz_evol_dispositions"].items():
    print(" ", k, "|", v, "=>", pf.render_promoted_row(v))
print()
txt = open(pf.REL_FRZ_CHECKLIST, encoding="utf-8").read()
for line in txt.splitlines():
    if line.startswith("| FRZ-EVOL-0") and ("FRZ-EVOL-06" in line or "FRZ-EVOL-07" in line or "FRZ-EVOL-01" in line):
        print("CHECKLIST ROW cells =", len(pf._table_cells(line)), "|", line[:160])
hdr = [l for l in txt.splitlines() if l.startswith("| ID") or l.startswith("| Req") ][:2]
print("HDR:", hdr)
reg = open(pf.REL_REGISTER, encoding="utf-8").read()
rows = pf.parse_register_rows(reg)
for od in pf.CLOSURE_OPERATOR_DECISIONS:
    print(od, rows.get(od))
print()
for s in ["Operator refuses B6 acceptance for EVOL-X", "B6 acceptance for EVOL-X cannot be authorized", "B6 acceptance for EVOL-X excluding everything", "B6 acceptance for EVOL-X except FRZ-EVOL-07", "B6 acceptance for EVOL-X without effect", "B6 acceptance for EVOL-X is NOT authorized", "B6 acceptance for EVOL-X isn't authorized", "B6 acceptance for EVOL-X is not authorized"]:
    print(repr(s), "->", repr(pf.scope_negation(s)))
