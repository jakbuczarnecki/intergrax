from atk_common import *
import json, subprocess
def git(repo,*a): return t._git(repo,*a)

# S3: dirty working tree for Gate-B-only inputs (not digest-bound)
c, cb = fresh("closed", "s3")
good = (c / pf.REL_FRZ_CHECKLIST).read_text(encoding="utf-8")
bad = good.replace("PASS — SCOPED", "PASS", 1)
assert bad != good
(c / pf.REL_FRZ_CHECKLIST).write_text(bad, encoding="utf-8"); t._commit(c, "widen a promoted FRZ row (committed)")
r = ev(c); show("S3a committed widened FRZ row (must FAIL B)", r)
(c / pf.REL_FRZ_CHECKLIST).write_text(good, encoding="utf-8")   # UNCOMMITTED repair
print("   git status:", git(c, "status", "--porcelain"))
r = ev(c); show("S3b same HEAD, uncommitted corrected checklist (HEAD still widened)", r)
# register
c, cb = fresh("closed", "s3r")
reg = (c / pf.REL_REGISTER).read_text(encoding="utf-8")
(c / pf.REL_REGISTER).write_text(reg.replace("SIGNED / APPROVED", "PENDING"), encoding="utf-8"); t._commit(c, "register: decisions pending (committed)")
show("S3c committed register PENDING", ev(c))
(c / pf.REL_REGISTER).write_text(reg, encoding="utf-8")
show("S3d uncommitted register restored", ev(c))
# ladder / governance json
c, cb = fresh("closed", "s3g")
g = t._load(c, pf.REL_GOVERNANCE); g2 = dict(g); g2["formal_b6_status"] = "PENDING"
t._dump(c, pf.REL_GOVERNANCE, g2); t._commit(c, "governance b6 pending (committed)")
show("S3e committed governance formal_b6_status PENDING", ev(c))
t._dump(c, pf.REL_GOVERNANCE, g)
show("S3f uncommitted governance restored", ev(c))
