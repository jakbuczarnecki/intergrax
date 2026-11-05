"""Session C: independent re-derivation of EVOL-M metrics at TES vs published FINAL bundle (read-only)."""
import json, subprocess, sys
from pathlib import Path
from testing_support.change_safety.discovery.closed_world import build_closed_world_inventory
from testing_support.change_safety.intake.git_change_set import rev_parse_head
from tests.qualification.evol_x.metrics.measure import collect_all
FINAL="bbf3e33bc62a05fd205e07f3d2e35a6d1d689ca1"
B="docs/project/maintainers/qualification/evol_x/evol_x_p03_independent_replay/evol_metrics.v1.json"
pub=json.loads(subprocess.run(["git","show",f"{FINAL}:{B}"],capture_output=True,check=True).stdout)
pubmap={r["metric_id"]:r for r in pub["readings"]}
head=rev_parse_head()
inv=build_closed_world_inventory(source_commit_sha=head)
mine={r.metric_id:r for r in collect_all(inv, head)}
out=[]; bad=0
for mid,pr in sorted(pubmap.items()):
    m=mine.get(mid)
    if m is None:
        out.append((mid,"NOT_RECOMPUTED(published status=%s)"%pr["status"])); 
        if pr["status"]!="NOT_MEASURED": bad+=1
        continue
    same = (m.numerator==pr["numerator"] and m.denominator==pr["denominator"] and (m.population_digest or None)==(pr["population_digest"] or None) and m.status.value==pr["status"])
    if not same: bad+=1
    out.append((mid,"MATCH" if same else "DIFF mine=%s/%s %s pub=%s/%s %s"%(m.numerator,m.denominator,m.status.value,pr["numerator"],pr["denominator"],pr["status"])))
for o in out: print(*o)
print("DIFFERENCES:",bad)
extra=set(mine)-set(pubmap); print("recomputed-not-published:",sorted(extra))
