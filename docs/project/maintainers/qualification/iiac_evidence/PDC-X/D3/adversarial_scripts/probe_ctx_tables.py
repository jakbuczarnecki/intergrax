import ast, pathlib, re
pat=re.compile(r"(CONTEXT|WINDOW|TOKEN_LIMIT|MAX_TOKENS|CTX)", re.I)
for r in ["intergrax","agents","applications"]:
  for p in pathlib.Path(r).rglob("*.py"):
    s=p.as_posix()
    if "/tests/" in s: continue
    try: t=ast.parse(p.read_text(encoding="utf-8"))
    except Exception: continue
    for n in ast.walk(t):
        tgt=None
        if isinstance(n,ast.Assign) and n.targets and isinstance(n.targets[0],(ast.Name,ast.Attribute)): tgt=n.targets[0]
        elif isinstance(n,ast.AnnAssign): tgt=n.target
        if tgt is None or not isinstance(getattr(n,'value',None),ast.Dict): continue
        name=tgt.id if isinstance(tgt,ast.Name) else tgt.attr
        d=n.value
        if len(d.keys)>=2 and pat.search(name) and all(isinstance(v,ast.Constant) and isinstance(v.value,int) and v.value>=1000 for v in d.values):
            print(s, n.lineno, name, len(d.keys))
