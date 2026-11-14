"""Certifier preflight addendum 3: round 5 (addendum 4 evidence integration) checked from origin/mcp-rc blobs.

Read-only against git. Writes nothing into the repository; the verifiers need real files, so the caller copies the
manifest and its four bound files from the pin-only commit into the worktree for the run and removes them afterwards.
"""
import hashlib, json, subprocess, sys

PUB, PIN, AUDB, OLD = "a1b3053b4", "fbe400f06", "origin/p7-00-baseline-audit", "d02c39950"
Q = "docs/project/maintainers/qualification/"
M = Q + "iiac_evidence/PDC-X/P7-00/addendum_4/iiac_audit_session_manifest.PDC-X.p7_00.addendum_4.json"


def sh(*a, raw=False):
    p = subprocess.run(a, capture_output=True)
    return p.stdout if raw else p.stdout.decode("utf-8", "replace")


h = lambda b: hashlib.sha256(b).hexdigest()
m = json.load(open(M, encoding="utf-8"))
print("pin-only commit:", sh("git", "rev-parse", PIN).strip(), "| publication commit:", sh("git", "rev-parse", PUB).strip())
print("## digests: claimed == bytes in the working copy (from the pin-only commit) | at the publication commit | on the auditor branch")
for dk, pk in {"audit_report_digest": "audit_report_path", "audit_scope_manifest_digest": "audit_scope_path",
               "test_matrix_digest": "test_matrix_path", "findings_digest": "findings_path"}.items():
    print(" ", dk, h(open(m[pk], "rb").read()) == m[dk], h(sh("git", "show", f"{PUB}:{m[pk]}", raw=True)) == m[dk],
          h(sh("git", "show", f"{AUDB}:{m[pk]}", raw=True)) == m[dk])
old = json.loads(sh("git", "show", f"{PUB}:{M}"))
aud = json.loads(sh("git", "show", f"{AUDB}:{M}"))
print("## pin-only diff, publication commit -> pin-only commit:")
print(" ", json.dumps({k: [old.get(k), m.get(k)] for k in sorted(set(old) | set(m)) if old.get(k) != m.get(k)}, indent=1))
print("## fields that differ from the auditor branch's manifest:", sorted(k for k in set(aud) | set(m) if aud.get(k) != m.get(k)))
print("## manifest SHA fields:", json.dumps({k: m[k] for k in ("audited_implementation_sha", "test_execution_sha", "audit_execution_sha",
      "evidence_publication_sha", "certification_publication_sha", "certifier_session_id")}, indent=1))
files = [l for l in sh("git", "diff", "--name-only", f"{PUB}~1", PUB).splitlines() if l]
diff = [f for f in files if sh("git", "rev-parse", f"{PUB}:{f}").strip() != sh("git", "rev-parse", f"{AUDB}:{f}").strip()]
print(f"## publication commit: files={len(files)} differing from the auditor branch={len(diff)} outside docs/={sum(1 for f in files if not f.startswith('docs/'))}")
print("## pin-only commit touches:", [l for l in sh("git", "diff", "--name-only", f"{PIN}~1", PIN).splitlines() if l])
print("## product / material governance / scripts diff", OLD, "..", PIN, ":", [l for l in sh("git", "diff", "--name-only", OLD, PIN, "--", "intergrax", "agents", "applications", "scripts",
      Q + "INTEGRAX_INTERNAL_INDEPENDENT_AGENT_CERTIFICATION_PROTOCOL.md").splitlines() if l])
print("## earlier P7-00 evidence changed", OLD, "..", PIN, "(outside addendum_4, the record, certifier):", [l for l in sh("git", "diff", "--name-only", OLD, PIN, "--", Q + "iiac_evidence/PDC-X/P7-00").splitlines()
      if l and "/addendum_4/" not in l and "EVIDENCE_INTEGRATION_RECORD" not in l and "/certifier/" not in l])
sys.stdout.flush()
