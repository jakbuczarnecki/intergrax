import sys, shutil, importlib.util, json, subprocess, os
from pathlib import Path
WT = Path(r"D:/Projekty/intergrax/.claude/worktrees/stoic-panini-83e1a9")
sys.path.insert(0, str(WT))
sys.path.insert(0, str(WT / "tests/unit/qualification"))
os.chdir(WT)
import test_evol_x_closure_preflight_b4 as t
pf = t.pf
ROOT = Path("D:/w4a/atk")

def make_base():
    dst = ROOT / "base" / "repo"
    if not dst.exists():
        built = t._build_base(dst)
        (ROOT / "base" / "built.json").write_text(json.dumps(built))
    return dst, json.loads((ROOT / "base" / "built.json").read_text())

def make_closed():
    dst = ROOT / "closed" / "repo"
    if not dst.exists():
        src, built = make_base()
        shutil.copytree(src, dst)
        b6 = t._close(dst, built)
        built = {**built, "b6": b6}
        (ROOT / "closed" / "built.json").write_text(json.dumps(built))
    return dst, json.loads((ROOT / "closed" / "built.json").read_text())

def fresh(kind, name):
    src, built = make_base() if kind == "base" else make_closed()
    dst = ROOT / "s" / name
    if dst.exists():
        shutil.rmtree(dst, onerror=lambda f, p, e: (os.chmod(p, 0o700), f(p)))
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst)
    return dst, built

def ev(repo):
    r = t._eval(repo)
    return r

def show(label, r, only=None):
    print(f"== {label}: gateA={'PASS' if r.gate_a_pass else 'FAIL'} gateB={'PASS' if r.gate_b_pass else 'FAIL'}")
    for e in r.gate_a[:6]: print("   A:", e[:230])
    if not r.gate_a_pass or True:
        for e in r.gate_b[:6]: print("   B:", e[:230])
