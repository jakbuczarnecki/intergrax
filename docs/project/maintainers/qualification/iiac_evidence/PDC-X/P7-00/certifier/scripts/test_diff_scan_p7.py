import subprocess, re, sys
def sh(*a):
    return subprocess.run(a, capture_output=True, text=True, encoding="utf-8", errors="replace").stdout
base, head = "2d3eceb6c", "ae5623efb"
files = [l.split("\t",1) for l in sh("git","diff","--name-status",base,head,"--","tests").splitlines()]
pats = {"tests": r"^\s*(async\s+)?def test_", "assert": r"^\s*assert\b", "raises": r"pytest\.raises", "skip": r"pytest\.(mark\.)?(skip|skipif)|importorskip|pytest\.skip\(", "xfail": r"xfail", "deselect": r"deselect|--ignore|-k "}
print(f"{'file':95} " + " ".join(f"{k}(b>h)" for k in pats))
tot = {k:[0,0] for k in pats}
for st, path in files:
    if st != "M": 
        print(("ADDED " if st=="A" else st+" ")+path); continue
    b = sh("git","show",f"{base}:{path}"); h = sh("git","show",f"{head}:{path}")
    row=[]
    for k,p in pats.items():
        cb = len(re.findall(p,b,re.M)); ch=len(re.findall(p,h,re.M))
        tot[k][0]+=cb; tot[k][1]+=ch
        row.append(f"{cb}>{ch}"+("*" if ch<cb else ""))
    print(f"{path[-95:]:95} "+" ".join(f"{r:>10}" for r in row))
print("TOTAL(M files):", {k:tuple(v) for k,v in tot.items()})
