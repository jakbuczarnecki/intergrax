"""P7-00 addendum 3: the lineage of the ollama regression (is B-REG-P and T-IGX-036-X the same item?).
Usage: python <this> <checkout_root>. Prints how default registration and the embedding resolve behave at that checkout."""
import json, sys
from pathlib import Path
root = Path(sys.argv[1]).resolve(); sys.path.insert(0, str(root))
# import order matters on the older checkouts (circular import through intergrax.runtime.events.store otherwise)
import intergrax.runtime.integrations.observability  # noqa: F401
from intergrax.integrations.registry import catalog as cat
from intergrax.integrations.registry.bootstrap import register_default_integrations, reset_default_integrations_state
from intergrax.integrations.contracts.base import IntegrationCategory as C
from intergrax.integrations.registry.factory import resolve
out = {"checkout": str(root)}
cat.clear_catalog(); reset_default_integrations_state()
try:
    register_default_integrations(preset="full"); out["bootstrap_full"] = "OK"
except Exception as exc:  # noqa: BLE001
    out["bootstrap_full"] = f"{type(exc).__name__}: {str(exc)[:140]}"
try:
    out["ollama_categories"] = [c.value for c in cat.get_entry("ollama").categories]
    out["ollama_row_count"] = cat.list_slugs().count("ollama")
except Exception as exc:  # noqa: BLE001
    out["ollama_row"] = f"{type(exc).__name__}: {exc}"
for cname in ("embedding_provider", "model_serving_runtime"):
    try:
        v = resolve(C(cname), "ollama"); out[f"resolve_{cname}"] = type(v).__name__
    except Exception as exc:  # noqa: BLE001
        out[f"resolve_{cname}"] = f"{type(exc).__name__}: {str(exc)[:110]}"
print(json.dumps(out, indent=1))
