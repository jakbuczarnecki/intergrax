"""Gate B semantic-row and staleness fixtures (synthetic repos only). Reproduction of the inline run documented in the dossier §5/§6 (G9-G11, S1, S2)."""
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, "D:/Projekty/intergrax")
from scripts.qualification import evol_x_closure_publication_preflight as pf
from tests.unit.qualification import test_evol_x_closure_preflight_b4 as T

ROOT = Path(tempfile.mkdtemp(prefix="rb", dir="C:/ia"))
BASE = ROOT / "base"
built = T._build_base(BASE)
CLOSED = ROOT / "closed"
shutil.copytree(BASE, CLOSED)
b6 = T._close(CLOSED, built)
n = [0]


def cl(src):
    n[0] += 1
    d = ROOT / f"w{n[0]}"
    shutil.copytree(src, d)
    return d


def ev(label, repo):
    r = pf.evaluate(repo, evidence_root=repo)
    print(
        f"{label}: A={r.gate_a_pass} B={r.gate_b_pass} A:{sorted({e.split(' ')[0] for e in r.gate_a})} "
        f"B:{[e[:110] for e in r.gate_b][:3]}"
    )


def setrow(w, frz, status, notes):
    rows = dict(T.PROMOTED_ROWS)
    rows[frz] = (status, notes)
    T._write(w, pf.REL_FRZ_CHECKLIST, T._checklist(rows))
    T._commit(w, "row")


w = cl(CLOSED)
setrow(w, "FRZ-EVOL-07", "PASS (EVOL-X SCOPED)", "LIM-05; qualification-layer only; not claimed -- BUT production rollback enforcement is verified and enforced")
ev("G9  FRZ-07 row text reversed semantically, keeps tokens", w)
w = cl(CLOSED)
setrow(w, "FRZ-EVOL-08", "PASS — SCOPED", "LIM-02; change-time diagnostics; M-09H not measured -> actually measured and passed")
ev("G10 FRZ-08 row claims M-09H measured, keeps tokens", w)
w = cl(CLOSED)
setrow(w, "FRZ-EVOL-01", "PASS — SCOPED", "LIM-03; complete platform-wide test ownership")
ev("G11 FRZ-01 row claims platform-wide ownership, keeps LIM-03", w)
w = cl(BASE)
T._write(w, "tests/qualification/evol_x/test_fixture_scope.py", "# changed after audit\n")
T._commit(w, "post-audit scope change")
ev("S1  audited scope changed after TES (A-11 expected)", w)
