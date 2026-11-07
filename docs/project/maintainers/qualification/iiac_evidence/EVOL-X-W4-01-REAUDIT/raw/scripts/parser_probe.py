import sys, os
sys.path.insert(0, r"D:/Projekty/intergrax/.claude/worktrees/stoic-panini-83e1a9"); os.chdir(r"D:/Projekty/intergrax/.claude/worktrees/stoic-panini-83e1a9")
from scripts.qualification import evol_x_closure_publication_preflight as pf
from datetime import datetime, timezone
def rec(scope, extra="", status="AUTHORIZED", sig=True):
    s = f"# R\n\nDecision ID: OD-EVOLX-08\nStage: EVOL-X\nPolicy version: {pf.POLICY_VERSION}\nDecision scope: {scope}\nDecision status: {status}\n{extra}"
    if sig: s += "Approved OD-EVOLX-08: operator @ 2026-11-06T10:00:00+00:00\n"
    return s
def chk(label, text, tokens=("EVOL-X","formal closure")):
    errs,_ = pf.operator_record_errors(text, decision_id="OD-EVOLX-08", scope_tokens=tokens, now=datetime(2026,11,6,12,tzinfo=timezone.utc))
    print(f"{label:70} -> {'ACCEPT' if not errs else 'REJECT: '+errs[0][:90]}")
chk("baseline ascii", rec("EVOL-X formal closure authorized"))
chk("Polish diacritics in scope (legit operator language)", rec("EVOL-X formal closure zatwierdzone przez operatora; zgoda na zamknięcie"))
chk("typographic em dash in scope", rec("EVOL-X formal closure \u2014 authorized"))
chk("scope: refuses ...", rec("EVOL-X formal closure: operator refuses"))
chk("scope: cannot ...", rec("EVOL-X formal closure cannot be authorized"))
chk("scope: except/excluding ...", rec("EVOL-X formal closure excluding all FRZ"))
chk("prose revocation line (col 0, no colon-key)", rec("EVOL-X formal closure authorized", extra="The operator revoked this authorization on 2026-11-07.\n"))
chk("second status line REVOKED", rec("EVOL-X formal closure authorized", extra="Decision status: REVOKED\n"))
chk("quoted revoked status", rec("EVOL-X formal closure authorized", extra="> Decision status: REVOKED\n"))
chk("revocation-type key 'Revoked: yes'", rec("EVOL-X formal closure authorized", extra="Revoked: yes\n"))
chk("CRLF record", rec("EVOL-X formal closure authorized").replace("\n","\r\n"))
chk("signature predates P2 (not_before checked elsewhere)", rec("EVOL-X formal closure authorized"))
# register
for st in ["SIGNED / APPROVED","Signed / Approved","signed/approved","SIGNED / APPROVED ✅","~~SIGNED / APPROVED~~","**SIGNED / APPROVED**","SIGNED / APPROVED (conditional)","DECIDED","AUTHORIZED","RATIFIED"]:
    e = pf.decision_status_errors(st,"2026-11-05")
    print(f"register status {st!r:40} -> {'ACCEPT' if not e else 'REJECT'}")
