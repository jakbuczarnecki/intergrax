from atk_common import *
import os, re
git = t._git
b, built = fresh("base", "s6c")
p2 = t._honest_p2(b)
print("P2 baseline blockers:", [x[:30] for x in json.loads(git(b,"show",f"{p2}:{pf.REL_CERT_RECORD}"))["open_blockers_for_closure"]])
# HEAD: delete every blocker (open and resolved) -> honest evaluator must reject
def wipe(c): c["open_blockers_for_closure"] = []; c["resolved_blockers_for_closure"] = []
t._mutate_json(b, pf.REL_CERT_RECORD, wipe, reindex=True)
r = pf.evaluate(b, evidence_root=b, p2_sha=p2); show("S6c-a HEAD without any blocker, honest history", r)
# forge replacement of P2 whose cert record also has no blockers (same date as P2)
pdate = git(b,"show","-s","--format=%cI",p2)
idx = str(ROOT/"s"/"s6c_index"); env = dict(os.environ, GIT_INDEX_FILE=idx, GIT_COMMITTER_DATE=pdate, GIT_AUTHOR_DATE=pdate)
def g(*a, inp=None): return subprocess.run(["git","-C",str(b),*a],capture_output=True,env=env,input=inp,check=True).stdout.decode().strip()
if os.path.exists(idx): os.remove(idx)
g("read-tree", p2)
rec = json.loads(git(b,"show",f"{p2}:{pf.REL_CERT_RECORD}")); rec["open_blockers_for_closure"] = []
blob = g("hash-object","-w","--stdin", inp=(json.dumps(rec, indent=2, ensure_ascii=False)+"\n").encode())
g("update-index","--add","--cacheinfo",f"100644,{blob},{pf.REL_CERT_RECORD}")
w = g("-c","user.email=x@y.z","-c","user.name=x","commit-tree",g("write-tree"),"-p",git(b,"rev-parse",f"{p2}^"),"-m","forged P2")
git(b,"replace",p2,w)
r = pf.evaluate(b, evidence_root=b, p2_sha=p2); show("S6c-b same HEAD + refs/replace/<P2> (blocker-free forged baseline)", r)
# mitigation check: same evaluation with replace objects disabled
os.environ["GIT_NO_REPLACE_OBJECTS"] = "1"
r = pf.evaluate(b, evidence_root=b, p2_sha=p2); show("S6c-c mitigation: GIT_NO_REPLACE_OBJECTS=1", r)
