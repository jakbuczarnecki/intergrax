from atk_common import *
import json, re
git = t._git
def ci(repo, msg):
    git(repo, "add", "-A"); git(repo, "-c","user.email=x@y.z","-c","user.name=x","commit","-m",msg); return git(repo,"rev-parse","HEAD")


# ---- S6: refs/replace on the pinned P2 commit ----------------------------------------------------------------
b, built = fresh("base", "s6d")
p2 = t._honest_p2(b)
# 1) widen LIM-01 at HEAD (committed) – honest evaluator rejects
def widen(m):
    i = t._find_limitation(m, "LIM-01"); m["limitations"][i] += t._WIDENING
widened_head_manifest = t._load(b, pf.REL_MANIFEST); widen(widened_head_manifest)
t._dump(b, pf.REL_MANIFEST, widened_head_manifest); t._reindex_cert(b); ci(b, "widen LIM-01 at HEAD")
r = pf.evaluate(b, evidence_root=b, p2_sha=p2); show("S6a widened HEAD, no replace ref (must FAIL A)", r)
# 2) forge a replacement for P2 carrying the widened manifest
cur = (b / pf.REL_MANIFEST).read_bytes()
p2_blob = git(b, "show", f"{p2}:{pf.REL_MANIFEST}")
m = pf._CERT_SHA_VALUE.search(p2_blob.encode("utf-8"))
old_val = re.search(rb'"certification_publication_sha"\s*:\s*"([^"]*)"', p2_blob.encode("utf-8")).group(1)
forged = pf._CERT_SHA_VALUE.sub(lambda mm: mm.group(1)+old_val+mm.group(2), cur, count=1)
# build tree: P2 tree with manifest replaced
import tempfile, os
tmp_index = str(ROOT / "s" / "s6d_index"); 
pdate = git(b,"show","-s","--format=%cI",p2)
env = dict(os.environ, GIT_INDEX_FILE=tmp_index, GIT_COMMITTER_DATE=pdate, GIT_AUTHOR_DATE=pdate)
def g(*a, inp=None):
    return subprocess.run(["git","-C",str(b),*a],capture_output=True,env=env,input=inp,check=True).stdout.decode().strip()
if os.path.exists(tmp_index): os.remove(tmp_index)
g("read-tree", p2)
blob = g("hash-object","-w","--stdin", inp=forged)
g("update-index","--add","--cacheinfo",f"100644,{blob},{pf.REL_MANIFEST}")
tree = g("write-tree")
parent = git(b,"rev-parse",f"{p2}^")
w = g("-c","user.email=x@y.z","-c","user.name=x","commit-tree",tree,"-p",parent,"-m","forged P2 replacement")
git(b, "replace", p2, w)
print("   replace refs:", git(b, "for-each-ref", "refs/replace"))
r = pf.evaluate(b, evidence_root=b, p2_sha=p2); show("S6b widened HEAD + refs/replace/<P2> -> forged P2", r)
# does a fresh clone carry the replace ref?
cl = ROOT / "s" / "s6d_clone"
if cl.exists(): shutil.rmtree(cl, onerror=lambda f,p,e:(os.chmod(p,0o700),f(p)))
subprocess.run(["git","clone","-q",str(b),str(cl)],check=True)
print("   replace refs in a plain `git clone`:", repr(subprocess.run(["git","-C",str(cl),"for-each-ref","refs/replace"],capture_output=True,text=True).stdout.strip()))
os.environ["GIT_NO_REPLACE_OBJECTS"] = "1"
r = pf.evaluate(b, evidence_root=b, p2_sha=p2); show("S6d mitigation GIT_NO_REPLACE_OBJECTS=1 on S6b state", r)
