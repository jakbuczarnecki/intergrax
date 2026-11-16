import ast, pathlib
roots=["intergrax","agents","applications"]
for r in roots:
    for p in pathlib.Path(r).rglob("*.py"):
        if "/tests/" in p.as_posix() or p.name.startswith("test_"): continue
        try: t=ast.parse(p.read_text(encoding="utf-8"))
        except Exception: continue
        for n in ast.walk(t):
            if isinstance(n,ast.ClassDef):
                ms={m.name for m in n.body if isinstance(m,(ast.FunctionDef,ast.AsyncFunctionDef))}
                if ms & {"stream_messages","stream_with_tools"}:
                    flags={k: k in ms for k in ("supports_streaming","supports_tool_streaming","supports_tools")}
                    bases=[ast.unparse(b) for b in n.bases]
                    print(p.as_posix(), n.name, bases, sorted(ms & {"stream_messages","stream_with_tools"}), flags)
