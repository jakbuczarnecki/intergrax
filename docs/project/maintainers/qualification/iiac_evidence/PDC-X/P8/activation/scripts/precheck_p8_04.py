"""CERT-PDC pre-checks for the PDC-X P8-04 rebind (read-only). Run from the checkout root.

Usage: python precheck_p8_04.py [C]   (C = certification_publication_sha; omit before C exists)
Imports the gate's own helpers (product_paths_changed_between, _findings_errors, resolve_evidence_path).
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO = Path.cwd()
sys.path.insert(0, str(REPO))
from scripts.governance import iiac_activation_gate as G_
from scripts.governance.iiac_common import git_merge_base_is_ancestor, product_paths_changed_between

Q = "docs/project/maintainers/qualification/"
GOV = "2ae1014116d353bc7742985e62f2bccc2bfac798"  # governance_implementation_sha (unchanged)
G = "969cbd964c679d563b02e55f065b452d46fdf90b"  # corrective / audited / test execution SHA
E = "a73d15ab341b30e9748aad03894f6faa75ad4e09"  # audit_execution_sha = evidence_publication_sha (merge of PR #597)
PREV_BIND = "a1b3053b49c28124a0abde679d733f21a597797f"  # evidence SHA of the binding in force (MCP-RC re-pin 2)
PIN0 = "2d3eceb6cf7e93dccd4d96919319a6c969b4444b"  # original P6 pin (union of all bindings)
C = sys.argv[1] if len(sys.argv) > 1 else ""


def git(*a):
    return subprocess.run(["git", *a], capture_output=True, text=True, check=False)


def rp(ref):
    return git("rev-parse", ref).stdout.strip()


def ok(name, cond, detail=""):
    print(f"[{'OK  ' if cond else 'FAIL'}] {name} {detail}")
    return bool(cond)


allok = True
head = rp("HEAD")
print("HEAD", head)
print("gate module:", G_.__file__)
allok &= ok("gate module resolves inside this checkout", Path(G_.__file__).resolve().is_relative_to(REPO.resolve()))

# 1. ancestry chain and SHA identity
for s in (GOV, G, E, PREV_BIND, PIN0) + ((C,) if C else ()):
    allok &= ok(f"{s[:9]} resolves to itself", rp(s) == s)
chain = [(GOV, G), (G, E), (E, head), (G, head), (PREV_BIND, G), (PIN0, G)]
if C:
    chain += [(E, C), (C, head)]
for a, b in chain:
    allok &= ok(f"ancestor {a[:9]} -> {b[:9]}", git_merge_base_is_ancestor(a, b, repo=REPO))
if C:
    allok &= ok("G != C", G != C)
    allok &= ok("E != C", E != C)
allok &= ok("origin/development contains E and G", git_merge_base_is_ancestor(E, "origin/development", repo=REPO))

# 2. product diffs (gate helper)
pairs = [(G, E, "G..E"), (E, head, "E..HEAD"), (G, head, "G..HEAD")]
if C:
    pairs.insert(1, (G, C, "G..C"))
for a, b, label in pairs:
    d = product_paths_changed_between(a, b, repo=REPO)
    allok &= ok(f"product diff {label} empty", d == [], f"n={len(d)} {d[:3]}")
delta = sorted(product_paths_changed_between(PREV_BIND, G, repo=REPO))
union = sorted(product_paths_changed_between(PIN0, G, repo=REPO))
print(f"INFO product paths a1b3053b4..G (delta audited since the binding in force): {len(delta)}")
for p in delta:
    print("  delta:", p)
print(f"INFO product paths 2d3eceb6c..G (union of all bindings up to G): {len(union)}")
under = {"intergrax": 0, "agents": 0, "applications": 0}
for p in delta:
    under[p.split("/")[0]] += 1
print("INFO delta by root:", under)

# 3. material governance paths
for ref in (E, G, PREV_BIND):
    names = git("diff", "--name-only", ref, "HEAD").stdout.split()
    mat = [n for n in names if n in G_.MATERIAL_GOVERNANCE_PATHS]
    if ref == PREV_BIND:
        print(f"INFO material governance paths changed {ref[:9]}..HEAD: {mat}")
    else:
        allok &= ok(f"no MATERIAL_GOVERNANCE_PATHS change {ref[:9]}..HEAD", mat == [], f"{mat}")
proto = git("log", "--format=%h", "-1", "HEAD", "--", Q + "INTEGRAX_INTERNAL_INDEPENDENT_AGENT_CERTIFICATION_PROTOCOL.md").stdout.strip()
allok &= ok("protocol last change precedes E", git_merge_base_is_ancestor(proto, E, repo=REPO), proto)
nondocs = [n for n in git("diff", "--name-only", G, "HEAD").stdout.split() if not n.startswith("docs/")]
print("INFO non-docs paths G..HEAD:", nondocs)

# 4. bound artifacts (kept IIAC-GOV-1 set): digests, bytes at E, path rules, ACT-N15
rec = json.loads(Path(Q + "IIAC_POLICY_ACTIVATION_RECORD.json").read_text(encoding="utf-8"))
want = {
    "audit_report_path": ("report", "c405c65faf53243a0511b0b9584a9f3abda4db08c09ee9b9266c86fe9f6b7a4e"),
    "audit_scope_path": ("scope", "e06ce433ac1c260390885e19cc60b0687be9d8908d78bd38ae4fb5f1102fe56f"),
    "test_matrix_path": ("matrix", "d5028f194bb7917d07e8df5264075952dcf8757503975042988caba6e89a240b"),
    "findings_path": ("findings", "df3648f926dad3199ff40c7c83c7fae4ed7a26bd1c67701a355bc7130073f831"),
}
for key, (kind, digest) in want.items():
    rel = rec[key]
    wt = Path(rel).read_bytes()
    allok &= ok(f"{key} sha256 == expected", hashlib.sha256(wt).hexdigest() == digest, hashlib.sha256(wt).hexdigest()[:12])
    allok &= ok(f"{key} bytes == bytes at E", wt == G_.git_show_bytes(REPO, E, rel))
    allok &= ok(f"{key} resolves under iiac_evidence (ACT-N14)", G_.resolve_evidence_path(REPO, rel, kind=kind, evidence_root=None).is_file())
allok &= ok("record digest field == bound report digest", rec["session_b_governance_audit_report_digest"] == want["audit_report_path"][1])
fj = json.loads(Path(rec["findings_path"]).read_text(encoding="utf-8"))
fe = G_._findings_errors(fj)
allok &= ok("findings register passes ACT-N15", fe == [], f"{fe} n={len(fj.get('findings', []))}")

# 5. global audit set: its own digests vs bytes at E (information for the limitations, not a gate input)
gm = json.loads(Path(Q + "iiac_evidence/PDC-X/global/iiac_audit_session_manifest.PDC-X.global.json").read_text(encoding="utf-8"))
for field, key in (
    ("audit_report_digest", "audit_report_path"),
    ("audit_scope_manifest_digest", "audit_scope_path"),
    ("test_matrix_digest", "test_matrix_path"),
    ("findings_digest", "findings_path"),
):
    rel = gm[key]
    at_e = hashlib.sha256(G_.git_show_bytes(REPO, E, rel)).hexdigest()
    good = at_e == gm[field]
    ok(f"global manifest {field} == bytes at E ({rel.rsplit('/', 1)[-1]})", good, f"at_E={at_e[:12]} manifest={gm[field][:12]}")
    allok &= good
allok &= ok("global manifest audited sha == G and audit_execution_sha == G", gm["audited_implementation_sha"] == G and gm["audit_execution_sha"] == G)
allok &= ok("global manifest verdict PASS_SCOPED", gm["verdict"] == "PASS_SCOPED")

# 6. activation text rules (6.1): not authored by the certifier
print("INFO decision-text rules of runbook 6.1: not applicable here; the OD file and the line belong to the activation delegate")
print("ALL_PRECHECKS_OK =", allok)
sys.exit(0 if allok else 1)
