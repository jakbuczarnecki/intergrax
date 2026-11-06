"""Independent adversarial experiments against a CLONE of the real EVOL-X evidence (never the canonical repo)."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

CLONE = Path("C:/ia/h")
PY = "D:/Projekty/intergrax/.venv/Scripts/python.exe"
QUAL = "docs/project/maintainers/qualification"
CERT = f"{QUAL}/iiac_evidence/EVOL-X/certification"
REC = f"{CERT}/EVOL_X_IIAC_CERTIFICATION_RECORD.json"
MAN = f"{CERT}/iiac_audit_session_manifest.EVOL-X.json"
IDX = f"{CERT}/CERTIFICATION_EVIDENCE_INDEX.json"
ACT = f"{QUAL}/IIAC_POLICY_ACTIVATION_RECORD.json"
GOV = f"{QUAL}/evol_x/EVOL_X_AUTO_B6_GOVERNANCE_STATUS.json"


def git(*a, check=True):
    p = subprocess.run(["git", "-c", "core.autocrlf=false", *a], cwd=CLONE, capture_output=True, text=True)
    if check and p.returncode:
        raise RuntimeError(p.stderr)
    return p.stdout.strip()


H0 = git("rev-parse", "HEAD")


def reset():
    git("reset", "-q", "--hard", H0)
    git("clean", "-fdq")


def rd(rel):
    return json.loads((CLONE / rel).read_text(encoding="utf-8"))


def wr(rel, data):
    (CLONE / rel).write_bytes((json.dumps(data, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))


def reindex():
    idx = rd(IDX)
    root = CLONE / CERT
    for name in list(idx["files"]):
        raw = (root / name).read_bytes()
        idx["files"][name] = {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
    wr(IDX, idx)


def commit(msg):
    git("add", "-A")
    git("-c", "user.email=a@b.c", "-c", "user.name=audit", "commit", "-q", "-m", msg)


def pf(*extra):
    p = subprocess.run([PY, "scripts/qualification/evol_x_closure_publication_preflight.py", "--json", *extra],
                       cwd=CLONE, capture_output=True, text=True)
    try:
        d = json.loads(p.stdout)
    except Exception:
        return {"raw": p.stdout[-400:], "err": p.stderr[-400:], "rc": p.returncode}
    d["rc"] = p.returncode
    return d


def show(label, d):
    a = d.get("gate_a_errors", [])
    b = [e for e in d.get("gate_b_errors", []) if not e.startswith("B-06")]
    print(f"--- {label}\n  mode={d.get('mode')} A_pass={d.get('gate_a_pass')} B_pass={d.get('gate_b_pass')} rc={d.get('rc')}")
    for e in a:
        print("  A:", e[:200])
    if a == [] or True:
        print("  B(non-FRZ) codes:", sorted({e.split(' ')[0] for e in b}))
    if "raw" in d:
        print("  RAW:", d)


def blockers(mut):
    rec = rd(REC)
    rec["open_blockers_for_closure"] = mut(rec["open_blockers_for_closure"])
    wr(REC, rec)
    reindex()
    commit("exp")


B4 = lambda s: s.startswith("B-4")
P3 = lambda s: s.startswith("B6 acceptance (P3)")

if __name__ == "__main__":
    which = sys.argv[1:]
    reset()
    show("X1a baseline real evidence", pf())

    reset(); blockers(lambda l: [x for x in l if not B4(x)])
    show("X1b B-4 line deleted, P3 line open", pf())

    reset(); blockers(lambda l: [x for x in l if not P3(x)])
    show("X1c P3 line deleted, B-4 open", pf())

    reset(); blockers(lambda l: [("B-4 RESOLVED - independently re-audited" if B4(x) else x) for x in l])
    show("X1d B-4 line rewritten as 'RESOLVED' (status string only)", pf())

    reset(); blockers(lambda l: [x for x in l if not B4(x) and not P3(x)])
    show("X1e B-4 + P3 lines simply DELETED (B-5 left open), no evidence reference of any kind", pf())
    d = pf("--gate", "b6-readiness"); print("  rc(b6-readiness)=", d.get("rc"))

    reset(); blockers(lambda l: [])
    show("X1f all blockers deleted", pf())


def batch2():
    import re
    def open_gate(l):
        return [x for x in l if not B4(x) and not P3(x)]

    # --- semantic scope widening on real evidence, with the blockers deleted so that only scope checks decide
    def widen(label, manifest_fn=None, rec_fn=None):
        reset()
        if manifest_fn:
            m = rd(MAN); manifest_fn(m); wr(MAN, m)
        rec = rd(REC)
        rec["open_blockers_for_closure"] = open_gate(rec["open_blockers_for_closure"])
        if rec_fn:
            rec_fn(rec)
        wr(REC, rec); reindex(); commit("exp")
        d = pf(); show(label, d)

    def app(lim, extra):
        def f(m):
            m["limitations"] = [(x + extra) if x.startswith(lim) else x for x in m["limitations"]]
        return f

    widen("X2a LIM-03 widened by APPENDED text (platform-wide test ownership complete)",
          app("LIM-03", "; HOWEVER platform-wide test ownership is established and complete for the whole platform"))
    widen("X2b LIM-05 appended: production rollback IS gated and enforced",
          app("LIM-05", "; update: production ActivationService.rollback enforcement is nevertheless in force"))
    widen("X2c LIM-02 appended: M-09H measured",
          app("LIM-02", "; correction: M-09H was in fact measured and passed"))
    widen("X2d LIM-04 appended: M-03 generalises to all arcs",
          app("LIM-04", "; M-03 discrimination power generalises to every arc"))
    widen("X2e FRZ-EVOL-07 disposition appended: rollback enforcement claimed",
          None, lambda r: r["frz_evol_dispositions"].__setitem__("FRZ-EVOL-07", r["frz_evol_dispositions"]["FRZ-EVOL-07"] + " ; production rollback enforcement IS claimed and verified"))
    widen("X2f FRZ-EVOL-01 disposition: unscoped wording + LIM-03 token kept",
          None, lambda r: r["frz_evol_dispositions"].__setitem__("FRZ-EVOL-01", "ELIGIBLE_PASS_SCOPED(LIM-03) full platform-wide unrestricted ownership"))
    widen("X2g cert verdict widened (PASS_SCOPED_... -> PASS_SCOPED but 'unrestricted')",
          None, lambda r: r.__setitem__("certification_verdict", "PASS_SCOPED_UNRESTRICTED_PLATFORM_WIDE"))

    # --- forged / tampered activation
    reset()
    a = rd(ACT); a["status"] = "NOT_IN_FORCE"; wr(ACT, a); commit("exp")
    show("X3a activation record committed as NOT_IN_FORCE (legacy fallback attempt)", pf())

    reset()
    a = rd(ACT); a["operator_decision_id"] = "FORGED-OD"; wr(ACT, a); commit("exp")
    show("X3b activation record committed with altered operator_decision_id", pf())

    reset()
    a = rd(ACT); a["session_b_verdict"] = "PASS"; a["handoff_digest"] = "0" * 64; wr(ACT, a)  # uncommitted
    show("X3c activation record uncommitted worktree tamper", pf())

    reset()
    a = rd(ACT); forged = CLONE.parent / "forged_act.json"; forged.write_text(json.dumps(a), encoding="utf-8")
    show("X3d --activation-record outside repo (copy of valid record)", pf("--activation-record", str(forged)))
    forged.unlink()

    reset()
    d = pf("--evidence-root", str(CLONE))
    show("X3e --evidence-root override on canonical repo", d)

    reset()
    del_a = rd(ACT); del_a.pop("status"); wr(ACT, del_a); commit("exp")
    show("X3f activation status key removed", pf())

    # --- evidence byte tamper
    reset()
    p = CLONE / f"{QUAL}/iiac_evidence/EVOL-X/audit/IIAC_EVOL_X_INDEPENDENT_AUDIT_DOSSIER.md"
    p.write_bytes(p.read_bytes() + b"\nforged tail\n")
    show("X4a audit dossier tampered (uncommitted)", pf())

    reset()
    p = CLONE / f"{QUAL}/iiac_evidence/EVOL-X/audit/IIAC_EVOL_X_INDEPENDENT_AUDIT_DOSSIER.md"
    p.write_bytes(p.read_bytes() + b"\nforged tail\n"); commit("exp")
    show("X4b audit dossier tampered and committed (digest not re-indexed)", pf())

    reset()
    f = rd(f"{QUAL}/iiac_evidence/EVOL-X/audit/EVOL-X_audit_findings.json")
    f["findings"].append({"id": "F-X-99", "class": "IN_SCOPE_BLOCKER", "severity": "HIGH", "disposition": "OPEN"})
    wr(f"{QUAL}/iiac_evidence/EVOL-X/audit/EVOL-X_audit_findings.json", f); commit("exp")
    show("X4c audit findings register mutated post-publication (blocker injected)", pf())

if __name__ == "__main__" and "b2" in sys.argv:
    batch2()
