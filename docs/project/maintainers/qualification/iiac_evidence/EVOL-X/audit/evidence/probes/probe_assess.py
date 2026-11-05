"""Session C Mode-C probes: exhaustive DM/WM invariants + extension-safety classification of all tracked files @ TES."""
import itertools, subprocess, collections, json
from testing_support.change_safety.assess import deployment_safety as D, workflow_evolution as W, extension_safety as E
from testing_support.change_safety.contracts.enums import CompatibilityAtChangeResult as C, DeploySafetyVerdict as V, WorkflowEvolutionDecision as WD

print("DeploySafetyVerdict:", [v.value for v in V], "| Compat:", [c.value for c in C], "| WD:", [w.value for w in WD])
# ---- P3: exhaustive DM invariants (operation incl. an invalid op)
viol = collections.Counter(); ex = {}
n = 0
for op, ev, cc, ps, fo, ip, dr in itertools.product(("rollback","roll_forward","weird"), (True,False), tuple(C), (True,False), (True,False), (True,False), (True,False)):
    f = D.DeployFacts(operation=op, evidence_present=ev, contract_compat=cc, persisted_schema_compatible=ps, durable_forward_only_applied=fo, inflight_pins=ip, drain_possible=dr)
    a = D.assess_deploy_safety(f); n += 1
    ok = a.verdict is V.ELIGIBLE
    checks = {
        "I1 eligible without evidence": ok and not ev,
        "I2 eligible rollback over forward-only durable migration": ok and op=="rollback" and fo,
        "I3 eligible with incompatible persisted schema": ok and not ps,
        "I4 eligible with non-COMPATIBLE contract": ok and cc is not C.COMPATIBLE,
        "I5 eligible for unknown operation": ok and op not in ("rollback","roll_forward"),
        "I6 eligible with inflight pins that cannot drain": ok and ip and not dr,
    }
    for k, bad in checks.items():
        if bad: viol[k] += 1; ex.setdefault(k, (f, a.matrix_id, a.verdict.value))
print(f"DM exhaustive: {n} combinations; invariant violations: {dict(viol) or 'NONE'}")
for k, v in ex.items(): print("  example", k, v)
# ---- P4: workflow matrix invariants
space = W.enumerate_input_space(); wviol = collections.Counter(); wex = {}
permissive = {WD.EXPLICITLY_UNSUPPORTED, WD.REJECT}
for c in space:
    d = W.classify_workflow_change(c).decision
    if c.has_inflight and c.adopt_in_place and c.fingerprint_changed and not c.migration_declared and d not in permissive:
        wviol["W1 in-place adoption of changed definition w/o migration not rejected"] += 1; wex.setdefault("W1", (c, d))
    if c.has_inflight and not c.pin_evidence and d not in permissive and c.fingerprint_changed:
        wviol["W2 in-flight changed definition without pin evidence not rejected"] += 1; wex.setdefault("W2", (c, d))
print(f"WM exhaustive: {len(space)} inputs; my-invariant violations: {dict(wviol) or 'NONE'}")
for k, v in wex.items(): print("  example", k, v)
print("WM decisions seen:", sorted({W.classify_workflow_change(c).decision.value for c in space}))
# ---- P2: extension-safety classification for every tracked file
files = subprocess.run(["git","ls-files"],capture_output=True,text=True,check=True).stdout.splitlines()
dist = collections.Counter(); gaps = collections.defaultdict(list)
for p in files:
    r = E.assess_extension_safety(changed_paths=frozenset([p]))
    top = p.split("/")[0]
    dist[(top, r.verdict.value)] += 1
    if r.verdict is E.ExtensionVerdict.EXTENSION_ONLY and (p.startswith(("intergrax/","scripts/","governance/",".github/")) or p in ("pyproject.toml","uv.lock")):
        gaps[top].append(p)
for k in sorted(dist): print(" ", k, dist[k])
print("EXTENSION_ONLY under non-extension roots (potential unflagged core-adjacent edits):", {k: len(v) for k, v in gaps.items()})
for k, v in gaps.items():
    print("  e.g.", k, v[:6])
    kinds = collections.Counter(x.rsplit(".",1)[-1] for x in v); print("   suffixes:", dict(kinds))
