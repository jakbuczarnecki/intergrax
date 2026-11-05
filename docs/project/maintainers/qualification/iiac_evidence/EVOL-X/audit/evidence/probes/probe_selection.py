"""Session C Mode-C probe: independent under-selection / discrimination probe at TES (read-only)."""
import json, random, sys, time
from collections import Counter
from pathlib import Path
from testing_support.change_safety.contracts.enums import SelectionStatus
from testing_support.change_safety.discovery.closed_world import build_closed_world_inventory
from testing_support.change_safety.graph.impact import assess_path_change_impact
from testing_support.change_safety.intake.git_change_set import rev_parse_head
from testing_support.change_safety.selection.selector import select_tests
from tests.qualification.evol_x.adapters.selection_inputs import build_selection_inputs
from tests.qualification.evol_x.metrics import measure as M

head = rev_parse_head()
t0 = time.time()
inv = build_closed_world_inventory(source_commit_sha=head)
idx = M._import_index()
rng = random.Random(20261105)
pools = {}
for p in sorted(idx):
    for root in ("intergrax/", "agents/", "applications/", "tests/unit/", "tests/qualification/", "platform_proofs/", "testing_support/change_safety/"):
        if p.startswith(root):
            pools.setdefault(root, []).append(p)
sample = []
for root, items in pools.items():
    rng.shuffle(items)
    sample += [(root, p) for p in items[:8]]
rows = []
for root, p in sample:
    changed = frozenset([p])
    impact = assess_path_change_impact(inv, changed_paths=changed)
    base = build_selection_inputs(impact, expected_head_sha=head)
    d = select_tests(base)
    sel = frozenset(n.node_id for n in d.selected) if d.selected else frozenset()
    closure = M._reverse_closure(changed)
    touches_self = any(x.startswith(M._SELF_SURFACE) for x in closure)
    universe = base.mandatory_node_ids | base.pop_a_node_ids | base.evol_manifest_node_ids | base.adjacent_node_ids
    rel = frozenset(n for n in universe if n in base.mandatory_node_ids or n.split("::",1)[0] in closure or (touches_self and n in base.evol_manifest_node_ids))
    missed = rel - sel
    extra = sel - rel
    rows.append({"root": root, "path": p, "status": d.status.value, "selected": len(sel), "oracle_relevant": len(rel), "under_selected": len(missed), "over_selected": len(extra)})
print(json.dumps({"head": head, "seconds": round(time.time()-t0,1), "n": len(rows)}))
st = Counter(r["status"] for r in rows)
print("status distribution:", dict(st))
print("distinct selected sizes:", sorted({r["selected"] for r in rows}))
under = [r for r in rows if r["status"]=="SELECTED" and r["under_selected"]>0]
print("SELECTED rows with under-selection (recall violations):", len(under))
for r in under[:10]: print("  ", r)
over = [r for r in rows if r["status"]=="SELECTED" and r["over_selected"]>0]
print("SELECTED rows with over-selection:", len(over), "of", sum(1 for r in rows if r['status']=='SELECTED'))
# unknown-path adversarial
for bogus in ["docs/zzz_unknown.md", "totally/unknown/file.bin", "intergrax/new_unclassified_module_xyz.py"]:
    impact = assess_path_change_impact(inv, changed_paths=frozenset([bogus]))
    d = select_tests(build_selection_inputs(impact, expected_head_sha=head))
    print("UNKNOWN", bogus, "->", impact.status.value, d.status.value, len(d.selected) if d.selected else 0)
Path(sys.argv[1]).write_text(json.dumps(rows, indent=1), encoding="utf-8")
