"""RB-2 activation PREP confirmations (read-only). Run from the checkout root."""
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
G0 = "969cbd964c679d563b02e55f065b452d46fdf90b"
E = "a73d15ab341b30e9748aad03894f6faa75ad4e09"
G2 = "c207e6ad3c01221e1e5867adaa21e0e5b038adc4"
DRAFT_MAN = (
    Q
    / "iiac_evidence/ARCH-FREEZE/RB-2/certifier/iiac_audit_session_manifest.ARCH-FREEZE-RB-2.certifier_draft.json"
)
PREP_HANDOFF = (
    Q
    / "iiac_evidence/ARCH-FREEZE/RB-2/activation/IIAC_SESSION_HANDOFF_RECORD_ARCH_FREEZE_RB2_ACTIVATION_PREP_DRAFT.md"
)
AMEND = Q / "IIAC_AMENDMENT_01_OPERATOR_DESIGNATED_SIGNING_SESSION.md"
PROTOCOL = Q / "INTEGRAX_INTERNAL_INDEPENDENT_AGENT_CERTIFICATION_PROTOCOL.md"
ALL = True


def git(*a):
    return subprocess.run(["git", *a], capture_output=True, text=True, check=False)


def ok(name: str, cond: bool, detail: str = "") -> None:
    global ALL
    ALL &= bool(cond)
    print(f"[{'OK  ' if cond else 'FAIL'}] {name} {detail}")


head = git("rev-parse", "HEAD").stdout.strip()
print("HEAD", head)

print("\n== Confirmation 1: delta and material governance (prep) ==")
for s in (G0, E, G2):
    ok(f"{s[:9]} resolves", git("rev-parse", s).stdout.strip() == s)
d = product_paths_changed_between(E, G2, repo=REPO)
ok("E..G2 product delta count == 76", len(d) == 76, f"n={len(d)}")
changed = G_._material_changes(REPO, E, head)
ok("no MATERIAL_GOVERNANCE_PATHS change E..HEAD at prep pin", changed == [], str(changed))

print("\n== Confirmation 2: certifier package still DRAFT (not activation-ready) ==")
man = json.loads(DRAFT_MAN.read_text(encoding="utf-8"))
ok("draft manifest not PASS", man.get("verdict") != "PASS", repr(man.get("verdict")))
ok("evidence_publication_sha pending", str(man.get("evidence_publication_sha", "")).startswith("PENDING"))
ok("certifier_session_id pending", str(man.get("certifier_session_id", "")).startswith("PENDING"))

print("\n== Confirmation 3: prep handoff exists; no binding manifest for RB-2 yet ==")
ok("prep handoff file exists", PREP_HANDOFF.is_file())
binding_rb2 = list((Q / "iiac_evidence/ARCH-FREEZE/RB-2/activation").glob("iiac_audit_session_manifest.*.json"))
ok("no RB-2 activation binding manifest (expected at prep)", binding_rb2 == [], str(binding_rb2))

print("\n== Confirmation 4: protocol still permits future delegated activation (not exercised) ==")
am = AMEND.read_text(encoding="utf-8")
proto = PROTOCOL.read_text(encoding="utf-8")
ok("Amendment 01 S-3 STRUCK", "**S-3 is STRUCK**" in am)
ok("protocol plain operator-form line documented", "operator @ <timestamp>" in proto)
ok("no protocol ban on commissioned writer", not re.search(r"(must not be (written|delegated))", proto, re.I))
ok("ancestor G0 -> G2 for future FCS tree", git_merge_base_is_ancestor(G0, G2, repo=REPO))

print("\nRESULT:", "ALL PREP CONFIRMATIONS OK" if ALL else "AT LEAST ONE PREP CONFIRMATION FAILED")
sys.exit(0 if ALL else 1)