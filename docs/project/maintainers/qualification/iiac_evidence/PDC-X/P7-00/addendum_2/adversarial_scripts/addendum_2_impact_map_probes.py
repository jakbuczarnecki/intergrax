"""P7-00 addendum 2: read-only probes of the CHANGED-SINCE impact map delta 19d552bee..324d9dba3 (auditor-owned).

Compares the impact map as shipped at the previous head (19d552bee, loaded from git) with the working tree
(324d9dba3 product tree): semantic JSON diff, per-file selection old vs new (including the entrypoint-touch and
invariant-floor parts of the real selection), manifest-id existence, entrypoint existence of every id used by the new
rules, generator/SSOT equality, sensitive-area coverage.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))
OLD = "19d552bee"
RES = "intergrax/platform/enterprise_audit/data/changed_since_impact_map.v1.json"


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout


from intergrax.platform.enterprise_audit.changed_since import _check_touched, invariant_floor  # noqa: E402
from intergrax.platform.enterprise_audit.impact_map import (  # noqa: E402
    PDC_DOMAIN_IMPACT_RULES,
    build_impact_map_from_manifest,
    check_ids_for_changed_path,
    fail_closed_prefixes,
    load_default_impact_map,
)
from intergrax.platform.enterprise_audit.manifest import load_default_manifest  # noqa: E402

checks = tuple(load_default_manifest())
manifest_ids = {c.check_id for c in checks}
old_raw = json.loads(git("show", f"{OLD}:{RES}"))
old_map = {k.rstrip("/"): frozenset(v) for k, v in old_raw["path_prefix_to_check_ids"].items()}
new_map = load_default_impact_map()
out: dict[str, object] = {}

# P1 semantic diff of the SSOT
removed_keys = sorted(set(old_map) - set(new_map))
added_keys = sorted(set(new_map) - set(old_map))
changed_keys = {
    k: {"dropped": sorted(old_map[k] - new_map[k]), "added": sorted(new_map[k] - old_map[k])}
    for k in old_map
    if k in new_map and old_map[k] != new_map[k]
}
out["P1_ssot_diff"] = {
    "old_keys": len(old_map),
    "new_keys": len(new_map),
    "removed_keys": removed_keys,
    "added_keys_count": len(added_keys),
    "added_keys": added_keys,
    "changed_keys": changed_keys,
}

# P2 ids that are not manifest ids, old vs new
out["P2_non_manifest_ids"] = {
    "old": sorted({i for v in old_map.values() for i in v} - manifest_ids),
    "new": sorted({i for v in new_map.values() for i in v} - manifest_ids),
    "manifest_size": len(manifest_ids),
}

# P3 per-file selection old vs new over every tracked file
files = git("ls-files").splitlines()
floor = invariant_floor(checks)
dropped_raw: list[dict] = []
dropped_effective: list[dict] = []
not_subset: list[dict] = []
changed_total = 0
for f in files:
    o, n = check_ids_for_changed_path(f, old_map), check_ids_for_changed_path(f, new_map)
    if o == n:
        continue
    changed_total += 1
    touched = {c.check_id for c in checks if _check_touched(c, frozenset({f}))}
    eff_old, eff_new = set(floor) | set(o) | touched, set(floor) | set(n) | touched
    if o - n:
        dropped_raw.append({"path": f, "dropped": sorted(o - n), "old": sorted(o), "new": sorted(n)})
    if eff_old - eff_new:
        dropped_effective.append({"path": f, "dropped": sorted(eff_old - eff_new)})
    if o and not (o <= n):
        not_subset.append({"path": f, "old": sorted(o), "new": sorted(n)})
out["P3_selection_old_vs_new"] = {
    "tracked_files": len(files),
    "files_whose_impact_ids_changed": changed_total,
    "files_with_dropped_impact_ids": len(dropped_raw),
    "files_with_effective_dropped_ids": len(dropped_effective),
    "dropped_impact_examples": dropped_raw[:60],
    "effective_dropped": dropped_effective[:60],
    "files_with_old_ids_not_subset_of_new": len(not_subset),
}

# P4 ids used by the new rules: exist in manifest, entrypoint exists
rule_ids = sorted({i for _, ids in PDC_DOMAIN_IMPACT_RULES for i in ids})
by_id = {c.check_id: c for c in checks}
ep_report: dict[str, object] = {}
for i in rule_ids:
    c = by_id.get(i)
    if c is None:
        ep_report[i] = "MISSING_IN_MANIFEST"
        continue
    ep = c.entrypoint.replace("\\", "/").split("::", 1)[0]
    ep_report[i] = {
        "entrypoint": c.entrypoint,
        "exists": (ROOT / ep).exists(),
        "fast_mandatory": bool(getattr(c, "fast_mandatory", False)),
    }
out["P4_rule_ids"] = {
    "count": len(rule_ids),
    "missing": [i for i, v in ep_report.items() if v == "MISSING_IN_MANIFEST"],
    "entrypoint_missing": [i for i, v in ep_report.items() if isinstance(v, dict) and not v["exists"]],
    "detail": ep_report,
}

# P5 generator equality
gen = build_impact_map_from_manifest(checks)
out["P5_ssot_equals_generator"] = {k: sorted(v) for k, v in gen.items()} == {k: sorted(v) for k, v in new_map.items()}


# P6 sensitive areas
def summarize(paths: list[str]) -> dict[str, object]:
    unmapped_old = [p for p in paths if not check_ids_for_changed_path(p, old_map)]
    unmapped_new = [p for p in paths if not check_ids_for_changed_path(p, new_map)]
    changed = [p for p in paths if check_ids_for_changed_path(p, old_map) != check_ids_for_changed_path(p, new_map)]
    return {
        "count": len(paths),
        "no_impact_ids_old": len(unmapped_old),
        "no_impact_ids_new": len(unmapped_new),
        "impact_ids_changed": len(changed),
        "unmapped_new_examples": unmapped_new[:15],
    }


product = ("intergrax/", "applications/", "agents/")
mcp_paths = [f for f in files if f.endswith(".py") and f.startswith(product) and "mcp" in f.lower()]
sens_tokens = ("/governance/", "/identity", "/auth/", "/authority/", "/policy/", "/tenant")
gov_paths = [f for f in files if f.endswith(".py") and f.startswith("intergrax/") and any(t in f for t in sens_tokens)]
rc23 = git("diff", "--name-only", "2d3eceb6c", "b0e19bf61", "--", "intergrax", "agents", "applications").split()
out["P6_sensitive_areas"] = {
    "mcp_named_python_files": summarize(mcp_paths),
    "governance_identity_authority_policy_python_files": summarize(gov_paths),
    "rc23_paths": summarize(rc23),
    "rc23_detail": {
        p: {
            "old_impact": sorted(check_ids_for_changed_path(p, old_map)),
            "new_impact": sorted(check_ids_for_changed_path(p, new_map)),
        }
        for p in rc23
    },
}

# P7 product areas without any mapped file in the new map
py_product = [f for f in files if f.endswith(".py") and f.startswith("intergrax/") and f.count("/") >= 2]
tops = sorted({"/".join(f.split("/")[:2]) for f in py_product})
out["P7_intergrax_top_level_without_any_mapped_file_new"] = [
    t for t in tops if not any(check_ids_for_changed_path(f, new_map) for f in py_product if f.startswith(t + "/"))
]
rt = [f for f in py_product if f.startswith("intergrax/runtime/") and f.count("/") >= 3]
out["P7b_runtime_subdirs_with_unmapped_files_new"] = sorted(
    {"/".join(f.split("/")[:3]) for f in rt if not check_ids_for_changed_path(f, new_map)}
)
out["P7c_runtime_subdirs_with_unmapped_files_old"] = sorted(
    {"/".join(f.split("/")[:3]) for f in rt if not check_ids_for_changed_path(f, old_map)}
)
out["P7d_unmapped_python_file_counts"] = {
    "intergrax_old": sum(1 for f in py_product if not check_ids_for_changed_path(f, old_map)),
    "intergrax_new": sum(1 for f in py_product if not check_ids_for_changed_path(f, new_map)),
    "intergrax_total": len(py_product),
}

# P8 fail-closed prefixes and own-file selection
own = (
    "intergrax/platform/enterprise_audit/impact_map.py",
    RES,
    "intergrax/platform/enterprise_audit/changed_since.py",
    "intergrax/platform/enterprise_audit/data/enterprise_audit_check_manifest.v1.json",
    "intergrax/platform/enterprise_audit/data/schemas/enterprise_audit_report.v1.schema.json",
)
out["P8_fail_closed_prefixes"] = list(fail_closed_prefixes())
out["P8_own_files_new"] = {p: sorted(check_ids_for_changed_path(p, new_map)) for p in own}
out["P8_own_files_old"] = {p: sorted(check_ids_for_changed_path(p, old_map)) for p in own}

print(json.dumps(out, indent=2, sort_keys=True, default=str))
