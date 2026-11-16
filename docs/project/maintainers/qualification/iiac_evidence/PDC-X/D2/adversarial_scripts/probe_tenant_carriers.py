import sys
sys.path.insert(0, ".")
from tests.qualification.tenant_x._replay_x_carrier_reconciliation import REPLAY_DELTA_CLASS_FIELD_SURFACE_IDS as K
from tests.qualification.tenant_x._inventory import raw_discovery_candidates
cands = raw_discovery_candidates()
field = [c.surface_id for c in cands if "::class.field." in c.surface_id and c.surface_id.endswith("class.field.tenant")]
print(len(cands), len(field))
import subprocess, json
# which carriers are in the head set but not in the FINAL set is not available; list those in files changed since the pinned head
import re
changed = subprocess.run(["git","diff","--name-only","5ccb72fccd4e3820ebdc9c18008afe6f7d59f558","HEAD","--","intergrax"],capture_output=True,text=True).stdout.split()
new = [s for s in field if s.split("::")[0] in set(changed) and s not in K]
print(len(new))
for s in sorted(new)[:60]: print(s)
