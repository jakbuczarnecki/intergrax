import sys, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from intergrax.applications._shared.product_manifest_registry import iter_strict_product_manifests
from intergrax.applications._shared.environment_wiring import wire_application_environment
res = {}
for pid, manifest in iter_strict_product_manifests():
    env = manifest.resolved_environment()
    try:
        wire_application_environment(manifest, env, conformance_check=False)
        res[pid] = "OK"
    except Exception as e:
        res[pid] = type(e).__name__ + ": " + str(e)[:600]
print(json.dumps(res, indent=1))
