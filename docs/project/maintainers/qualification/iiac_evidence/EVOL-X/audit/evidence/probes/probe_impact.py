"""Session C Mode-C probe P1: tool impact graph vs independent AST reverse-import closure on real modules @ TES."""
import json, random, collections
from testing_support.change_safety.discovery.closed_world import build_closed_world_inventory
from testing_support.change_safety.graph.impact import assess_path_change_impact
from testing_support.change_safety.intake.git_change_set import rev_parse_head
from tests.qualification.evol_x.metrics import measure as M
head = rev_parse_head()
inv = build_closed_world_inventory(source_commit_sha=head)
idx = M._import_index()
by_dotted = {M._dotted(p): p for p in idx}
rng = random.Random(7771)
roots = ("intergrax/contracts/", "intergrax/runtime/", "intergrax/decision_system/", "intergrax/agent_distribution/", "agents/", "applications/", "testing_support/change_safety/", "intergrax/platform/")
pool = [p for p in sorted(idx) if p.startswith(roots) and not p.endswith("__init__.py")]
rng.shuffle(pool); sample = pool[:60]
under_total = over_total = closure_total = 0; rows = []; CAT = collections.Counter(); SUITE_EX = []
for p in sample:
    impact = assess_path_change_impact(inv, changed_paths=frozenset([p]))
    tool = set()
    for n in impact.affected:
        k, v = n.node_id.split(":",1)
        if k == "module": tool.add(v)
        elif k == "suite" and v.endswith(".py"): tool.add(M._dotted(v))
    clos = {M._dotted(x) for x in M._reverse_closure(frozenset([p]))}
    under = clos - tool; over = tool - clos
    closure_total += len(clos); under_total += len(under); over_total += len(over)
    prod_under = sorted(u for u in under if not u.startswith("tests."))
    test_under = sorted(u for u in under if u.startswith("tests."))
    for u in test_under:
        leaf = u.rsplit(".",1)[-1]
        cat = "suite(test_*)" if leaf.startswith("test_") else ("conftest" if leaf=="conftest" else "helper/support")
        CAT[cat] += 1
        if cat == "suite(test_*)" and len(SUITE_EX) < 8: SUITE_EX.append((p, u))
    rows.append((p, impact.status.value, len(clos), len(tool), len(under), len(over), prod_under[:4], len(prod_under), len(test_under)))
bad = [r for r in rows if r[7] > 0]
print("TOTAL missed production(non-tests) modules:", sum(r[7] for r in rows), "| missed test modules:", sum(r[8] for r in rows))
print("paths with missed production modules:", len(bad))
print("sampled:", len(rows), "status:", dict(collections.Counter(r[1] for r in rows)))
print("oracle closure total:", closure_total, "tool-missed (under-detected):", under_total, "tool-extra:", over_total)
print("paths with under-detection:", len(bad))
for r in bad[:15]: print("  ", r[0], "prod_missed=", r[7], r[6])

print("missed test-module categories:", dict(CAT))
for e in SUITE_EX: print("  suite example:", e)
# total test modules in closures (for fraction)
