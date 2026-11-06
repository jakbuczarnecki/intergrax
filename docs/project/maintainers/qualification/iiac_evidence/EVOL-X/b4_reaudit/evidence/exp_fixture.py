"""Independent fixture experiments (synthetic repos only). Reuses the implementer's *builders* but supplies its own scenarios/assertions."""
import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, "D:/Projekty/intergrax")
from tests.unit.qualification import test_evol_x_closure_preflight_b4 as T  # noqa: E402
from scripts.qualification import evol_x_closure_publication_preflight as pf  # noqa: E402

ROOT = Path(tempfile.mkdtemp(prefix="ra", dir="C:/ia"))
BASE = ROOT / "base"
built = T._build_base(BASE)
CLOSED = ROOT / "closed"
shutil.copytree(BASE, CLOSED)
b6 = T._close(CLOSED, built)
RES = []
CNT = [0]


def clone(src):
    CNT[0] += 1
    dst = ROOT / f"w{CNT[0]}"
    shutil.copytree(src, dst)
    return dst


def ev(label, repo, expect_a=None, expect_b=None):
    r = pf.evaluate(repo, evidence_root=repo)
    ok = (expect_a is None or r.gate_a_pass == expect_a) and (expect_b is None or r.gate_b_pass == expect_b)
    RES.append(ok)
    codes_a = sorted({e.split(' ')[0] for e in r.gate_a})
    codes_b = sorted({e.split(' ')[0] for e in r.gate_b})
    print(f"{'OK  ' if ok else 'DIFF'} {label}: mode={r.mode} A={r.gate_a_pass} B={r.gate_b_pass} Acodes={codes_a} Bcodes={codes_b}"
          f"{'  expected A=%s B=%s' % (expect_a, expect_b) if not ok else ''}")
    return r


def mut_rec(repo, fn, reindex=True):
    d = T._load(repo, pf.REL_CERT_RECORD)
    fn(d)
    T._dump(repo, pf.REL_CERT_RECORD, d)
    if reindex:
        T._reindex_cert(repo)
    T._commit(repo, "exp")


B4_TXT = "B-4 legacy closure preflight requires independent corrective implementation and re-audit"
P3_TXT = "B6 acceptance (P3), FRZ promotion (P4) and formal closure (P5) not authorized in this publication"

print("== Blocker-state transition (isolated fixtures) ==")
w = clone(BASE); mut_rec(w, lambda d: d.__setitem__("open_blockers_for_closure", [B4_TXT, P3_TXT, "B-5 OD pending"]))
ev("T1 B-4 open, P3 open", w, False, False)
w = clone(BASE); mut_rec(w, lambda d: d.__setitem__("open_blockers_for_closure", [B4_TXT, "B-5 OD pending"]))
ev("T2 B-4 open, P3 authorized", w, False, False)
w = clone(BASE); mut_rec(w, lambda d: d.__setitem__("open_blockers_for_closure", [P3_TXT, "B-5 OD pending"]))
ev("T3 B-4 resolved, P3 NOT authorized", w, False, False)
w = clone(BASE)
r = ev("T4 B-4 resolved + P3 authorized (B-5 open) -> Gate A PASS, Gate B must FAIL", w, True, False)
w = clone(BASE); mut_rec(w, lambda d: d.__setitem__("open_blockers_for_closure", ["B-4 RESOLVED by independent re-audit", P3_TXT]))
ev("T5 B-4 'status string only' rewrite", w, False, False)
w = clone(BASE); mut_rec(w, lambda d: d.__setitem__("open_blockers_for_closure", ["unprefixed open issue"]))
ev("T5c unprefixed blocker blocks", w, False, False)
w = clone(BASE); mut_rec(w, lambda d: d.__setitem__("open_blockers_for_closure", "not-a-list"))
ev("T5d blockers field malformed (string)", w, False, False)
w = clone(BASE); mut_rec(w, lambda d: d.pop("open_blockers_for_closure"))
ev("T5e blockers field absent altogether -> ?", w, None, None)

print("== Gate B ==")
ev("G1 complete positive closure fixture", CLOSED, True, True)
r = pf.evaluate(CLOSED, evidence_root=CLOSED)
print("   closure fixture has FORMAL_CLOSURE_PUBLICATION_SHA?", pf.ladder_sha(pf.parse_ladder_active((CLOSED / pf.REL_LADDER).read_text(encoding='utf-8')), "FORMAL_CLOSURE_PUBLICATION_SHA") or "none")
ev("G2 readiness only (no B6 SHA) -> Gate B FAIL", clone(BASE), True, False)

# B6 SHA variants on closed fixture
w = clone(CLOSED)
g = T._load(w, pf.REL_GOVERNANCE); g["b6_acceptance_publication_sha"] = built["cert_pub"]; T._dump(w, pf.REL_GOVERNANCE, g)
c = T._load(w, pf.REL_CERT_RECORD); c["b6_acceptance_publication_sha"] = built["cert_pub"]; T._dump(w, pf.REL_CERT_RECORD, c); T._reindex_cert(w)
l = (w / pf.REL_LADDER).read_text(encoding="utf-8").replace(b6, built["cert_pub"]); T._write(w, pf.REL_LADDER, l)
T._commit(w, "b6=cert_pub")
ev("G3 B6 SHA == certification publication SHA", w, None, False)

w = clone(CLOSED)
g = T._load(w, pf.REL_GOVERNANCE); g["b6_acceptance_publication_sha"] = built["tes"]; T._dump(w, pf.REL_GOVERNANCE, g)
c = T._load(w, pf.REL_CERT_RECORD); c["b6_acceptance_publication_sha"] = built["tes"]; T._dump(w, pf.REL_CERT_RECORD, c); T._reindex_cert(w)
l = (w / pf.REL_LADDER).read_text(encoding="utf-8").replace(b6, built["tes"]); T._write(w, pf.REL_LADDER, l)
T._commit(w, "b6=tes")
ev("G4 B6 SHA == historical (pre-certification) commit", w, None, False)

# arbitrary later commit that does NOT contain governance 'ACCEPTED' (a no-op later commit used as b6)
w = clone(BASE)
T._write(w, "noop.txt", "x\n"); noop = T._commit(w, "unrelated later commit")
T._dump(w, pf.REL_GOVERNANCE, T._governance(formal_b6_status="ACCEPTED", b6_acceptance_policy_version=pf.POLICY_VERSION, b6_acceptance_publication_sha=noop))
T._write(w, pf.REL_LADDER, T._ladder(built["tes"], built["final"], built["tes"], b6=noop))
c = T._load(w, pf.REL_CERT_RECORD); c["b6_acceptance"] = "ISSUED"; c["b6_acceptance_publication_sha"] = noop; c["open_blockers_for_closure"] = []
T._dump(w, pf.REL_CERT_RECORD, c); T._reindex_cert(w)
T._write(w, pf.REL_FRZ_CHECKLIST, T._checklist(T.PROMOTED_ROWS)); T._write(w, pf.REL_REGISTER, T._register(decided=True))
T._commit(w, "bind")
ev("G5 B6 SHA = unrelated no-op commit (phase-1 blob lacks ACCEPTED)", w, None, False)

# phase-1 sets b6 flags in the SAME commit as everything (self-consistent single commit): should the SHA be a parent commit?
# Variant: 'accepted' flag published by an arbitrary committer, no authorization artefact needed
w = clone(BASE)
T._dump(w, pf.REL_GOVERNANCE, T._governance(formal_b6_status="ACCEPTED", b6_acceptance_policy_version=pf.POLICY_VERSION))
c = T._load(w, pf.REL_CERT_RECORD); c["b6_acceptance"] = "ISSUED"; T._dump(w, pf.REL_CERT_RECORD, c); T._reindex_cert(w)
p1 = T._commit(w, "phase1 by anyone")
T._dump(w, pf.REL_GOVERNANCE, T._governance(formal_b6_status="ACCEPTED", b6_acceptance_policy_version=pf.POLICY_VERSION, b6_acceptance_publication_sha=p1))
T._write(w, pf.REL_LADDER, T._ladder(built["tes"], built["final"], built["tes"], b6=p1))
c = T._load(w, pf.REL_CERT_RECORD); c["b6_acceptance_publication_sha"] = p1; c["open_blockers_for_closure"] = []
T._dump(w, pf.REL_CERT_RECORD, c); T._reindex_cert(w)
T._write(w, pf.REL_FRZ_CHECKLIST, T._checklist(T.PROMOTED_ROWS)); T._write(w, pf.REL_REGISTER, T._register(decided=True))
T._commit(w, "bind")
ev("G6 (info) B6 'acceptance' = flag flips + empty blocker list; no authorization artefact referenced", w, None, None)

# FORMAL_CLOSURE ladder SHA handling
w = clone(CLOSED)
T._write(w, pf.REL_LADDER, (w / pf.REL_LADDER).read_text(encoding="utf-8").replace("| INTERNAL_CLOSURE_STATUS", f"| FORMAL_CLOSURE_PUBLICATION_SHA | `{built['cert_pub']}` | bad |\n| INTERNAL_CLOSURE_STATUS"))
T._commit(w, "closure sha before b6")
ev("G7 FORMAL_CLOSURE_PUBLICATION_SHA older than B6 -> must FAIL", w, None, False)
w = clone(CLOSED)
head = T._git(w, "rev-parse", "HEAD")
T._write(w, pf.REL_LADDER, (w / pf.REL_LADDER).read_text(encoding="utf-8").replace("| INTERNAL_CLOSURE_STATUS", f"| FORMAL_CLOSURE_PUBLICATION_SHA | `{head}` | ok |\n| INTERNAL_CLOSURE_STATUS"))
T._commit(w, "closure sha after b6")
ev("G8 FORMAL_CLOSURE_PUBLICATION_SHA = prior commit after B6 -> may PASS", w, None, True)

print("== Register status semantics (B-07) ==")
for status in ("SIGNED / APPROVED", "NOT APPROVED", "NOT CONFIRMED", "REVOKED (previously APPROVED)", "REJECTED - not approved", "UNSIGNED", "APPROVED - WITHDRAWN", "CONFIRMED? NO"):
    print(f"  decision_recorded({status!r}, '2026-11-06') ->", pf.decision_recorded(status, "2026-11-06"))

print("== Post-P2 mutation of certified manifest (limitations not in auditor draft) ==")
def drop_cert_limits(m):
    keep = [x for x in m["limitations"] if x[:6] in {"LIM-01","LIM-02","LIM-03","LIM-04","LIM-05","LIM-06","LIM-07"}]
    m["limitations"] = keep
w = clone(BASE)
def mm(fn):
    m = T._load(w, pf.REL_MANIFEST); fn(m); T._dump(w, pf.REL_MANIFEST, m); T._reindex_cert(w); T._commit(w, "mut manifest")
mm(drop_cert_limits)
ev("M1 certifier-authored limitations dropped from manifest after P2 (F-C-02 / Evidence-only lines)", w, True, False)

print("== Role separation tamper ==")
w = clone(BASE); mm2 = T._load(w, pf.REL_MANIFEST); mm2["certifier_session_id"] = mm2["auditor_session_id"]; T._dump(w, pf.REL_MANIFEST, mm2)
c = T._load(w, pf.REL_CERT_RECORD); c["certifier_session_id"] = mm2["auditor_session_id"]; T._dump(w, pf.REL_CERT_RECORD, c); T._reindex_cert(w); T._commit(w, "x")
ev("R1 certifier session == auditor session", w, False, False)

print("SUMMARY: all-as-expected =", all(RES), "| fixtures at", ROOT)
