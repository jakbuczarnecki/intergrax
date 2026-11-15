import os
for k in list(os.environ):
    if k.endswith("_API_KEY"): os.environ.pop(k)
from intergrax.integrations.registry.bootstrap import register_default_integrations
from intergrax.integrations.registry.catalog import list_slugs, get_entry
from intergrax.integrations.registry.factory import resolve
register_default_integrations(preset="full")
slugs=list_slugs()
print("slugs", len(slugs), "dups", len(slugs)-len(set(slugs)))
rows=0
for s in sorted(set(slugs)):
    e=get_entry(s)
    if len(e.categories)>1:
        for c in e.categories:
            rows+=1
            try:
                r=resolve(c, s)
                print(f"OK    {s:22} {c.value:24} -> {type(r).__name__}")
            except Exception as ex:
                print(f"FAIL  {s:22} {c.value:24} -> {type(ex).__name__}: {str(ex)[:110]}")
print("multi-category resolves", rows)
