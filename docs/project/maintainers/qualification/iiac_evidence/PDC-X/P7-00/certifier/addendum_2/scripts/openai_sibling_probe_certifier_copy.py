"""P7-00 addendum 3: does the sibling shared-slug embedding provider (openai) have the defect that ollama had?

Usage: python addendum_3_openai_sibling_probe.py <checkout_root>
Read-only; a dummy placeholder value for OPENAI_API_KEY is set in-process only to get past the managed-retrieval config
check; no client call is made (the factory only constructs the integration object).
"""
import json, os, sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(root))
from intergrax.integrations.contracts.base import IntegrationCategory as C
from intergrax.integrations.registry import catalog as cat
from intergrax.integrations.registry.bootstrap import register_default_integrations, reset_default_integrations_state
from intergrax.integrations.registry.factory import resolve

cat.clear_catalog(); reset_default_integrations_state(); register_default_integrations(preset="full")
row = cat.get_entry("openai")
res = {"row_categories": [c.value for c in row.categories], "row_factory": getattr(row.factory, "__qualname__", "?")}
for label, env in (("no_key", None), ("dummy_key", "placeholder-not-a-real-key")):
    if env is None:
        os.environ.pop("OPENAI_API_KEY", None)
    else:
        os.environ["OPENAI_API_KEY"] = env
    for cname in ("embedding_provider", "managed_retrieval"):
        try:
            v = resolve(C(cname), "openai")
            res[f"{label}|{cname}"] = {"ok": f"{type(v).__module__}.{type(v).__qualname__}"}
        except Exception as exc:  # noqa: BLE001
            res[f"{label}|{cname}"] = {"error": f"{type(exc).__name__}: {str(exc)[:140]}"}
print(json.dumps(res, indent=1))
