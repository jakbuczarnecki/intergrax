"""RB-2 activation PREP pre-checks (read-only). Run from the checkout root.

Usage: python precheck_rb2_activation_prep.py
Uses the gate's product_paths_changed_between and draft certifier manifest pins.
"""
import json
import subprocess
import sys
from pathlib import Path

REPO = Path.cwd()
sys.path.insert(0, str(REPO))
from scripts.governance import iiac_activation_gate as G_
from scripts.governance.iiac_common import git_merge_base_is_ancestor, product_paths_changed_between

Q = REPO / "docs/project/maintainers/qualification"
G0 = "969cbd964c679d563b02e55f065b452d46fdf90b"
E = "a73d15ab341b30e9748aad03894f6faa75ad4e09"
G2 = "c207e6ad3c01221e1e5867adaa21e0e5b038adc4"
DELTA = Q / "iiac_evidence/ARCH-FREEZE/RB-2/certifier/delta_files.txt"
DRAFT_MAN = (
    Q
    / "iiac_evidence/ARCH-FREEZE/RB-2/certifier/iiac_audit_session_manifest.ARCH-FREEZE-RB-2.certifier_draft.json"
)
PRIOR_BIND = Q / "iiac_evidence/PDC-X/P8/activation/iiac_audit_session_manifest.IIAC-ACT-N11-PDC-X-P8-04.json"


def git(*a):
    return subprocess.run(["git", *a], capture_output=True, text=True, check=False)


def rp(ref: str) -> str:
    return git("rev-parse", ref).stdout.strip()


def ok(name: str, cond: bool, detail: str = "") -> bool:
    print(f"[{'OK  ' if cond else 'FAIL'}] {name} {detail}")
    return bool(cond)


allok = True
head = rp("HEAD")
print("HEAD", head)
print("gate module:", G_.__file__)
allok &= ok("gate module resolves inside this checkout", Path(G_.__file__).resolve().is_relative_to(REPO.resolve()))

for s in (G0, E, G2):
    allok &= ok(f"{s[:9]} resolves to itself", rp(s) == s)

for a, b in ((G0, E), (E, G2), (G0, G2)):
    allok &= ok(f"ancestor {a[:9]} -> {b[:9]}", git_merge_base_is_ancestor(a, b, repo=REPO))

delta_mechanical = sorted(product_paths_changed_between(E, G2, repo=REPO))
delta_listed = DELTA.read_text(encoding="utf-8").splitlines()
delta_listed = [ln.strip() for ln in delta_listed if ln.strip()]
allok &= ok("mechanical E..G2 product delta count == 76", len(delta_mechanical) == 76, f"n={len(delta_mechanical)}")
allok &= ok("delta_files.txt line count == 76", len(delta_listed) == 76, f"n={len(delta_listed)}")
allok &= ok(
    "delta_files.txt matches mechanical E..G2",
    delta_listed == delta_mechanical,
    "mismatch" if delta_listed != delta_mechanical else "",
)

man = json.loads(DRAFT_MAN.read_text(encoding="utf-8"))
allok &= ok("certifier draft manifest_status", man.get("manifest_status") == "CERTIFIER_DRAFT_NOT_A_PASS")
allok &= ok("certifier draft verdict DRAFT", man.get("verdict") == "DRAFT")
allok &= ok("certifier audited_implementation_sha == G2", man.get("audited_implementation_sha") == G2)

prior = json.loads(PRIOR_BIND.read_text(encoding="utf-8"))
allok &= ok("prior binding verdict PASS", prior.get("verdict") == "PASS")
allok &= ok("prior audited sha == G0", prior.get("audited_implementation_sha") == G0)

reg = (Q / "ARCH_FREEZE_OPERATOR_DECISION_REGISTER.md").read_text(encoding="utf-8")
allok &= ok("OD-AF-12 OPEN in operator register", "OD-AF-12" in reg and "RB-2 activation line" in reg)

# Informational: activation still stale vs G2 product tree until RB-2 completes
prod_head = product_paths_changed_between(G0, head, repo=REPO)
print(f"INFO product paths G0..HEAD @ prep pin: n={len(prod_head)} (ACT-N11 stale until RB-2)")
mat = G_._material_changes(REPO, E, head)
print(f"INFO material governance paths changed E..HEAD: {mat}")
allok &= ok("no edit to IIAC_POLICY_ACTIVATION_RECORD in this worktree diff", not any(
    ln.endswith("IIAC_POLICY_ACTIVATION_RECORD.json")
    for ln in git("diff", "--name-only", "HEAD").stdout.splitlines()
))

print("ALL_PREP_PRECHECKS_OK =", allok)
sys.exit(0 if allok else 1)