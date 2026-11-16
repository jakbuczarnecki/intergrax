import sys, subprocess, shutil
from pathlib import Path
sys.path.insert(0, ".")
from scripts.qualification import mcp_rc_mutation as h
S = Path("D:/Projekty/_wt/intergrax/_scratch/mcp-rc-cert-m4m5")
head = subprocess.check_output(["git","rev-parse","HEAD"], text=True).strip()
subprocess.run(["git","worktree","add","--detach",str(S),head], check=True, capture_output=True)
try:
    for m in h.MUTANTS:
        if m.mutant_id[:2] not in ("M4","M5"): continue
        assert h._replace_once(S/m.path, m.old, m.new), m.mutant_id
        p = subprocess.run([sys.executable,"-m","pytest",*m.tests,"-q","--tb=short","-x","-p","no:warnings"], cwd=S, capture_output=True, text=True, encoding="utf-8", errors="replace")
        out = p.stdout + p.stderr
        print("=====", m.mutant_id, "exit", p.returncode)
        lines = out.splitlines()
        keep = [l for l in lines if l.startswith(("E ", "FAILED", "___")) or "AssertionError" in l or " passed" in l or " failed" in l]
        print("\n".join(keep[:25]))
        subprocess.run(["git","checkout","--",m.path], cwd=S, capture_output=True)
finally:
    subprocess.run(["git","worktree","remove","--force",str(S)], capture_output=True)
    shutil.rmtree(S, ignore_errors=True)
