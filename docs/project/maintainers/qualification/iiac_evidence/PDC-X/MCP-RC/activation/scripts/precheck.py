"""Activation certifier independent pre-checks (read-only). Run from the checkout root."""
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path.cwd()
sys.path.insert(0, str(REPO))
from scripts.governance import iiac_activation_gate as G
from scripts.governance.iiac_common import product_paths_changed_between, git_merge_base_is_ancestor

Q = "docs/project/maintainers/qualification/"
PIN = "2d3eceb6cf7e93dccd4d96919319a6c969b4444b"
GOV = "2ae1014116d353bc7742985e62f2bccc2bfac798"
AUD = "324d9dba37cb54e1b74a79a05148d2dcc05e6d09"
EVID = "29d66ae3fcc7a06a919d7e07541740841e8abfe8"
CERTPUB = "7801dcb1293e31ee0c483184efd59c01a2dc606b"
MCPLAST = "b0e19bf61b346e6941b2e190b703c6922e708b34"
OD = Q + "IIAC_OD_ACTIVATION_MCP_RC_2026-11-13.md"


def git(*a):
    return subprocess.run(["git", *a], capture_output=True, text=True, check=False)


def rp(ref):
    return git("rev-parse", ref).stdout.strip()


def ok(name, cond, detail=""):
    print(f"[{'OK ' if cond else 'FAIL'}] {name} {detail}")
    return cond


allok = True
head = rp("HEAD")
print("HEAD", head)
print("origin/mcp-rc", rp("origin/mcp-rc"))

# 1. pin chain
chain = [(GOV, AUD), (AUD, EVID), (EVID, head), (CERTPUB, head), (PIN, AUD), (AUD, head), (EVID, CERTPUB)]
for a, b in chain:
    allok &= ok(f"ancestor {a[:9]} -> {b[:9]}", git_merge_base_is_ancestor(a, b, repo=REPO))
for s in (GOV, AUD, EVID, CERTPUB, PIN):
    allok &= ok(f"{s[:9]} resolves to itself", rp(s) == s)

# 158 union
union = product_paths_changed_between(PIN, AUD, repo=REPO)
union_head = product_paths_changed_between(PIN, head, repo=REPO)
union_evid = product_paths_changed_between(PIN, EVID, repo=REPO)
allok &= ok("ACT-N11 union pin..audited tree", len(union) == 158, f"count={len(union)}")
allok &= ok("union pin..HEAD equals pin..audited", sorted(union) == sorted(union_head), f"count={len(union_head)}")
allok &= ok("union pin..evidence equals", sorted(union) == sorted(union_evid), f"count={len(union_evid)}")
# audited sha -> head, audited -> evid empty product diff
for a, b, label in ((AUD, head, "audited..HEAD"), (AUD, EVID, "audited..evidence_publication"), (EVID, head, "evidence..HEAD")):
    d = product_paths_changed_between(a, b, repo=REPO)
    allok &= ok(f"product diff {label} empty", d == [], f"n={len(d)}")
# MCP-RC 23 paths
mcp = [p for p in union if "mcp" in p.lower() and (p.startswith("applications/") and p.endswith("mcp/server.py") or "/_shared/mcp_" in p or p in ("intergrax/contracts/mcp_server_definition.py", "intergrax/tools/mcp_result_bounds.py"))]
scope = json.loads(Path(Q + "iiac_evidence/PDC-X/MCP-RC/audit_scope.json").read_text(encoding="utf-8"))
mcp_ref = sorted(scope["product_paths_delta_vs_base"])
allok &= ok("23 MCP-RC paths equal audit_scope product_paths_delta_vs_base", len(mcp_ref) == 23 and set(mcp_ref) <= set(union), f"n={len(mcp_ref)}")
d = git("diff", "--name-only", MCPLAST, "HEAD", "--", *mcp_ref).stdout.split()
allok &= ok("23 MCP-RC paths byte-identical to b0e19bf61 at HEAD", d == [], f"diff={d}")
print("non-MCP:", len(set(union) - set(mcp_ref)))
# MATERIAL_GOVERNANCE_PATHS unchanged since audit_execution_sha (and since pin)
for ref in (EVID, PIN):
    names = git("diff", "--name-only", ref, "HEAD").stdout.split()
    mat = [n for n in names if n in G.MATERIAL_GOVERNANCE_PATHS]
    allok &= ok(f"no MATERIAL_GOVERNANCE_PATHS change {ref[:9]}..HEAD", mat == [], f"{mat}")
# all non-docs changes between audited tree and HEAD
print("non-docs diff audited..HEAD:", git("diff", "--name-only", AUD, "HEAD").stdout.split().__len__(), "files total")
print([n for n in git("diff", "--name-only", AUD, "HEAD").stdout.split() if not n.startswith("docs/")])

# 2. operator decision rules
text = Path(OD).read_text(encoding="utf-8")
lines = text.splitlines()
sig = [l for l in lines if l.startswith("Approved IIAC policy activation")]
allok &= ok("exactly one signature line", len(sig) == 1, repr(sig))
line = sig[0]
m = G._SIGNATURE.fullmatch(line)
allok &= ok("signature regex fullmatch", bool(m), m.group(1) if m else "")
print("line bytes:", line.encode("utf-8"))
allok &= ok("no trailing space/CR on the line in the file bytes", b"\r" not in Path(OD).read_bytes() and line == line.strip())
op_id = "IIAC-OD-MCP-RC-ACT-N11-2026-11-13"
digest = "c405c65faf53243a0511b0b9584a9f3abda4db08c09ee9b9266c86fe9f6b7a4e"
rec = {"operator_decision_id": op_id, "operator_activation_line": line}
errs = G._operator_errors(rec, text, digest)
allok &= ok("gate _operator_errors empty", errs == [], str(errs))
errs = G.closure_conflicts(rec, text)
allok &= ok("ACT-N18 closure_conflicts empty (record+text)", errs == [], str(errs))
for needle in ("________________", "NOT SIGNED", "Decision status: PENDING"):
    allok &= ok(f"forbidden string absent: {needle!r}", needle not in text)
name = Path(OD).name
allok &= ok("file name rule", name.startswith("IIAC_OD_ACTIVATION_") and "PACKET" not in name.upper() and "DRAFT" not in name.upper())
allok &= ok("'Approved:' line absent (repo-forbidden for me)", not any(l.startswith("Approved:") for l in lines))

# 3. digests of working-tree bytes
paths = {
    "audit_report_path": "iiac_evidence/IIAC-GOV-1/certification/IIAC_ACT01_INDEPENDENT_REAUDIT_DOSSIER.md",
    "audit_scope_path": "iiac_evidence/ENT-AUDIT-X/act_n11_recovery/audit_scope.json",
    "test_matrix_path": "iiac_evidence/ENT-AUDIT-X/act_n11_recovery/test_matrix.txt",
    "findings_path": "iiac_evidence/HIST-P5-2026-11-09/HIST_P5_REACTIVATION_FINDINGS.json",
}
want = {
    "audit_report_path": "c405c65faf53243a0511b0b9584a9f3abda4db08c09ee9b9266c86fe9f6b7a4e",
    "audit_scope_path": "e06ce433ac1c260390885e19cc60b0687be9d8908d78bd38ae4fb5f1102fe56f",
    "test_matrix_path": "d5028f194bb7917d07e8df5264075952dcf8757503975042988caba6e89a240b",
    "findings_path": "df3648f926dad3199ff40c7c83c7fae4ed7a26bd1c67701a355bc7130073f831",
}
for k, rel in paths.items():
    full = Q + rel
    wt = hashlib.sha256(Path(full).read_bytes()).hexdigest()
    committed = G.git_show_bytes(REPO, EVID, full)
    allok &= ok(f"{k} worktree sha256 == expected", wt == want[k], wt)
    allok &= ok(f"{k} bytes == bytes at evidence_publication_sha", Path(full).read_bytes() == committed)
    allok &= ok(f"{k} under iiac_evidence", G.resolve_evidence_path(REPO, full, kind={"audit_report_path": "report", "audit_scope_path": "scope", "test_matrix_path": "matrix", "findings_path": "findings"}[k], evidence_root=None).is_file())
fj = json.loads(Path(Q + paths["findings_path"]).read_text(encoding="utf-8"))
fe = G._findings_errors(fj)
allok &= ok("ACT-N15 findings register passes", fe == [], f"{fe} n={len(fj.get('findings', []))}")
print("ALL_PRECHECKS_OK =", allok)
sys.exit(0 if allok else 1)
