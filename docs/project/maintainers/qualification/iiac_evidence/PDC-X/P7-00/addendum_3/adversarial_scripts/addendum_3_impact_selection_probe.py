"""P7-00 addendum 3: which manifest checks does a CHANGED-SINCE run select for the delta path, and for the eight
applications/*/mcp/server.py modules (F-41 status at this head)? Read-only. Usage: python <this> <checkout_root> <out_json>"""
import glob, json, sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(root))
from intergrax.platform.enterprise_audit import changed_since as cs  # noqa: E402
from intergrax.platform.enterprise_audit.impact_map import check_ids_for_changed_path, load_default_impact_map  # noqa: E402
from intergrax.platform.enterprise_audit.manifest import load_default_manifest  # noqa: E402

delta = "intergrax/integrations/providers/embedding_provider/ollama/register.py"
mcp8 = sorted(p.replace("\\", "/").replace(str(root).replace("\\", "/") + "/", "") for p in glob.glob(str(root / "applications/*/mcp/server.py")))
res = {"checkout": str(root), "delta_path": delta, "mcp_server_modules_found": mcp8}
try:
    prefixes = load_default_impact_map()
except Exception as exc:  # noqa: BLE001
    prefixes = None
    res["load_error"] = f"{type(exc).__name__}: {exc}"
if prefixes is not None:
    res["delta_impact_ids"] = sorted(check_ids_for_changed_path(delta, prefixes))
    res["mcp_modules_impact_ids"] = {p: sorted(check_ids_for_changed_path(p, prefixes)) for p in mcp8}
    res["mcp_modules_without_any_rule"] = [p for p in mcp8 if not check_ids_for_changed_path(p, prefixes)]
checks = load_default_manifest()
res["manifest_checks"] = len(checks)
res["invariant_floor_count"] = len(cs.invariant_floor(checks))
for since in ("324d9dba37cb54e1b74a79a05148d2dcc05e6d09", "2d3eceb6cf7e93dccd4d96919319a6c969b4444b"):
    try:
        sel = cs.select_checks_for_changed_since(checks, repo_root=root, since_sha=since)
        res[f"selection_since_{since[:9]}"] = {"count": len(sel), "ids": sorted(sel)}
    except Exception as exc:  # noqa: BLE001
        res[f"selection_since_{since[:9]}"] = {"error": f"{type(exc).__name__}: {str(exc)[:300]}"}
Path(sys.argv[2]).write_text(json.dumps(res, indent=2), encoding="utf-8")
print(json.dumps({k: (v if not isinstance(v, dict) or "ids" not in v else {"count": v["count"]}) for k, v in res.items()}, indent=1)[:2500])

