"""P7-00 addendum 3: repeated and overriding default registration with the wrapped ollama row. Usage: <this> <checkout_root>"""
import json, sys
from pathlib import Path
root = Path(sys.argv[1]).resolve(); sys.path.insert(0, str(root))
from intergrax.integrations.contracts.base import IntegrationCategory as C
from intergrax.integrations.registry import catalog as cat
from intergrax.integrations.registry.bootstrap import register_default_integrations, reset_default_integrations_state
from intergrax.integrations.registry.factory import resolve
out = {}
def snap(label):
    row = cat.get_entry("ollama")
    res = {}
    for c in (C.EMBEDDING_PROVIDER, C.MODEL_SERVING_RUNTIME):
        try:
            res[c.value] = type(resolve(c, "ollama")).__name__
        except Exception as exc:  # noqa: BLE001
            res[c.value] = f"{type(exc).__name__}: {str(exc)[:90]}"
    rev = cat.read_catalog_revision_under_lock()
    out[label] = {"categories": [c.value for c in row.categories], "rows": cat.list_slugs().count("ollama"), "resolve": res, "generation": rev.generation, "factory": getattr(row.factory, "__qualname__", "?")}
cat.clear_catalog(); reset_default_integrations_state()
register_default_integrations(preset="full"); snap("first_full")
try:
    register_default_integrations(preset="full"); snap("second_full_no_override_is_noop")
except Exception as exc:  # noqa: BLE001
    out["second_full_no_override"] = f"{type(exc).__name__}: {str(exc)[:120]}"
try:
    register_default_integrations(preset="full", override=True); snap("third_full_override")
except Exception as exc:  # noqa: BLE001
    out["third_full_override"] = f"{type(exc).__name__}: {str(exc)[:160]}"
print(json.dumps(out, indent=1))
