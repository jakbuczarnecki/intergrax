"""Activation-delegate confirmations 1, 3 and 4 for PDC-X P8-04 (read-only). Run from the checkout root.

Usage: python confirm_p8_04_act.py
Confirmation 1 uses the gate's own helper product_paths_changed_between and the gate's MATERIAL_GOVERNANCE_PATHS.
Confirmation 2 (verify_iiac_policy_activation preconditions, verify_iiac_role_separation) is a separate logged call.
"""
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path.cwd()
sys.path.insert(0, str(REPO))
from scripts.governance import iiac_activation_gate as G_
from scripts.governance.iiac_common import git_merge_base_is_ancestor, product_paths_changed_between

Q = REPO / "docs/project/maintainers/qualification"
G = "969cbd964c679d563b02e55f065b452d46fdf90b"
E = "a73d15ab341b30e9748aad03894f6faa75ad4e09"
C = "fd2ad28efde6216be38194b020188d116264adb6"
BINDING = Q / "iiac_evidence/PDC-X/P8/activation/iiac_audit_session_manifest.IIAC-ACT-N11-PDC-X-P8-04.json"
HANDOFF = Q / "iiac_evidence/PDC-X/P8/activation/IIAC_SESSION_HANDOFF_RECORD_PDC_X_P8_04_ACTIVATION.md"
AUDIT = Q / "platform_domain_completeness/PDC_INDEPENDENT_AUDIT.md"
DELEG = Q / "platform_domain_completeness/operator_decisions/DELEGATION-PDC-P8-ACT.md"
AMEND = Q / "IIAC_AMENDMENT_01_OPERATOR_DESIGNATED_SIGNING_SESSION.md"
PROTOCOL = Q / "INTEGRAX_INTERNAL_INDEPENDENT_AGENT_CERTIFICATION_PROTOCOL.md"
ALL = True


def git(*a):
    return subprocess.run(["git", *a], capture_output=True, text=True, check=False)


def ok(name, cond, detail=""):
    global ALL
    ALL &= bool(cond)
    print(f"[{'OK  ' if cond else 'FAIL'}] {name} {detail}")


head = git("rev-parse", "HEAD").stdout.strip()
print("HEAD", head)
print("gate module:", G_.__file__)
ok("gate module resolves inside this checkout", Path(G_.__file__).resolve().is_relative_to(REPO.resolve()))

print("\n== Confirmation 1: ACT-N11 product delta and material governance paths ==")
for s in (G, E, C):
    ok(f"{s[:9]} resolves to itself", git("rev-parse", s).stdout.strip() == s)
for a, b in ((G, E), (E, C), (C, head), (G, head)):
    ok(f"ancestor {a[:9]} -> {b[:9]}", git_merge_base_is_ancestor(a, b, repo=REPO))
for a, b, label in ((G, E, "G..E"), (G, C, "G..C"), (E, head, "E..HEAD"), (G, head, "G..HEAD")):
    d = product_paths_changed_between(a, b, repo=REPO)
    ok(f"product diff {label} empty (gate helper product_paths_changed_between)", d == [], f"n={len(d)} {d[:3]}")
print("MATERIAL_GOVERNANCE_PATHS (gate):")
for p in sorted(G_.MATERIAL_GOVERNANCE_PATHS):
    print("  ", p)
ok("six material governance paths in the gate", len(G_.MATERIAL_GOVERNANCE_PATHS) == 6)
for ref, label in ((E, "E"), (G, "G")):
    changed = G_._material_changes(REPO, ref, head)
    ok(f"no MATERIAL_GOVERNANCE_PATHS change {label}..HEAD (gate helper _material_changes)", changed == [], str(changed))

print("\n== Confirmation 3: global audit published, binding manifest verdict ==")
audit = AUDIT.read_text(encoding="utf-8")
m = re.search(r"^### 15\.2 Final verdict\s*\n\s*\n\*\*(\w+)\*\*", audit, re.M)
ok("PDC_INDEPENDENT_AUDIT.md section 15.2 final verdict is PASS_SCOPED", bool(m) and m.group(1) == "PASS_SCOPED", repr(m.group(1) if m else None))
ok("audit file is in origin/development (published)", git("cat-file", "-e", "origin/development:" + AUDIT.relative_to(REPO).as_posix()).returncode == 0)
man = json.loads(BINDING.read_text(encoding="utf-8"))
ok("binding manifest verdict is PASS", man.get("verdict") == "PASS", repr(man.get("verdict")))
ok("manifest audited_implementation_sha == G", man.get("audited_implementation_sha") == G)
ok("manifest audit_execution_sha == evidence_publication_sha == E", man.get("audit_execution_sha") == E == man.get("evidence_publication_sha"))
ok("manifest certification_publication_sha == C", man.get("certification_publication_sha") == C)
ok("manifest audit_report_digest is c405c65f...7a4e", str(man.get("audit_report_digest")).startswith("c405c65f") and str(man.get("audit_report_digest")).endswith("7a4e"))
print("binding manifest sha256", hashlib.sha256(BINDING.read_bytes()).hexdigest())
print("handoff sha256        ", hashlib.sha256(HANDOFF.read_bytes()).hexdigest())
ok("manifest digest equals the certifier report", hashlib.sha256(BINDING.read_bytes()).hexdigest() == "77fa2babd02df6fbbd2e0d5b34413a769e069ac293e485daf49ec9a779dc476f")
ok("handoff digest equals the certifier report", hashlib.sha256(HANDOFF.read_bytes()).hexdigest() == "a36b2ac35eaf0782f4d40ef9b76a4dd745b0c4e70d839e2df2a7fa819b1fd121")

print("\n== Confirmation 4: protocol and Amendment 01 permit a delegated activation line ==")
deleg = DELEG.read_text(encoding="utf-8")
am = AMEND.read_text(encoding="utf-8")
proto = PROTOCOL.read_text(encoding="utf-8")
ok("DELEGATION-PDC-P8-ACT part (c) names the plain form written by the activation delegate", "Approved IIAC policy activation IIAC-2026-11-05.1: operator @ <timestamp>" in deleg and "activation delegate writes the activation line in the plain form" in deleg)
ok("Amendment 01 section 0: S-3 STRUCK, IIAC policy activation not excluded", "**S-3 is STRUCK**" in am and "including the delegation record, IIAC policy activation and amendments of this rule" in am)
ok("Amendment 01 S-2: plain form allowed if the commission names it", "may write the plain form if the commission names it" in am)
ok("Amendment 01 R-6: the operator may commission a separate session to certify in the operator name", "commission a separate agent session to certify in the operator" in am)
ok("Amendment 01 6a(a): who writes the line is invisible to the gate", "is invisible to the gate" in am)
ok("protocol Policy activation item 3 states the plain operator-form line", "operator @ <timestamp>" in proto)
ok("protocol text has no ban on a commissioned writer", not re.search(r"(personally|must not be (written|delegated)|not delegable)", proto, re.I))

print("\nRESULT:", "ALL CONFIRMATIONS OK" if ALL else "AT LEAST ONE CONFIRMATION FAILED")
sys.exit(0 if ALL else 1)
