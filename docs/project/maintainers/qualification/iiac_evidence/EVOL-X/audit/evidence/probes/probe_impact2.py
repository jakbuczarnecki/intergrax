import json, random, collections
from testing_support.change_safety.discovery.closed_world import build_closed_world_inventory
from testing_support.change_safety.graph.impact import assess_path_change_impact
from testing_support.change_safety.intake.git_change_set import rev_parse_head
from tests.qualification.evol_x.metrics import measure as M
head = rev_parse_head()
inv = build_closed_world_inventory(source_commit_sha=head)
p = "intergrax/runtime/capacity/queue_depth.py"
imp = assess_path_change_impact(inv, changed_paths=frozenset([p]))
print("node-id prefix kinds:", collections.Counter(n.node_id.split(":",1)[0] for n in imp.affected))
for n in imp.affected[:12]: print(" ", n.node_id, n.kind, n.rule_id)
clos = M._reverse_closure(frozenset([p]))
print("closure files:", sorted(clos))
