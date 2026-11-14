"""P7-00 addendum 4: mutation sample on the Tier-3 root impact rule (auditor-owned).

usage: python addendum_4_map_mutation_sample.py <scratch-checkout>
For each mutant: edit impact_map.py, regenerate the SSOT JSON with scripts/enterprise_audit/build_impact_map.py --emit-ssot
(so the shipped-JSON-equals-generator test is not the only killer), run the targeted tests once, restore the files with git.
Targeted tests: pdc_x/p6_impact (coverage test, tier-3 rule test), XD-12 and the ent_audit_x unit test of the impact map.
A mutant is KILLED only by a failure that is not red in the control run. One pytest process at a time.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
IMPACT = root / "intergrax/platform/enterprise_audit/impact_map.py"
SSOT = root / "intergrax/platform/enterprise_audit/data/changed_since_impact_map.v1.json"
TESTS = [
    "tests/qualification/pdc_x/p6_impact",
    "tests/qualification/pdc_x/xd/test_xd_12_evol_ent_audit.py",
    "tests/qualification/ent_audit_x/unit/test_changed_since_impact_map.py",
]
env = dict(os.environ)
env["PYTHONPATH"] = str(root).replace("\\", "/") + os.pathsep + env.get("PYTHONPATH", "")
env["PYTHONUTF8"] = "1"
env["PYTHONDONTWRITEBYTECODE"] = "1"
START = "PDC_TIER3_ROOT_IMPACT_RULES: tuple["
IDS_OPEN = '        "applications",\n        (\n'


def run(cmd: list[str], timeout: int = 900) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=root, env=env, capture_output=True, text=True, timeout=timeout)


def restore() -> None:
    run(["git", "checkout", "--", str(IMPACT.relative_to(root)), str(SSOT.relative_to(root))])


def once(text: str, old: str, new: str) -> str:
    assert text.count(old) == 1, f"mutation anchor not unique or missing: {old!r}"
    return text.replace(old, new)


def tier3_block(text: str) -> tuple[int, int]:
    start = text.index(START)
    end = text.index("\n)\n", start) + 3
    return start, end


def edit_tier3(text: str, fn) -> str:  # noqa: ANN001
    start, end = tier3_block(text)
    return text[:start] + fn(text[start:end]) + text[end:]


def replace_ids(block: str, ids: tuple[str, ...]) -> str:
    head, rest = block.split(IDS_OPEN, 1)
    tail = rest[rest.index("        ),\n") :]
    body = "".join(f'            "{i}",\n' for i in ids)
    return head + IDS_OPEN + body + tail


MUTANTS = {
    "N1_drop_tier3_table_from_generator": lambda t: once(
        t, "(*PDC_DOMAIN_IMPACT_RULES, *PDC_TIER3_ROOT_IMPACT_RULES)", "(*PDC_DOMAIN_IMPACT_RULES,)"
    ),
    "N2_reroute_tier3_root_to_doc_check": lambda t: edit_tier3(t, lambda b: replace_ids(b, ("ENT-A-DOC-01",))),
    "N3_shrink_tier3_root_to_one_id": lambda t: edit_tier3(t, lambda b: replace_ids(b, ("ENT-A-BND-01",))),
    "N4_tier3_prefix_with_trailing_slash": lambda t: edit_tier3(t, lambda b: once(b, '"applications",', '"applications/",')),
    "N5_unknown_id_in_tier3_root": lambda t: edit_tier3(
        t, lambda b: replace_ids(b, ("ENT-A-BND-01", "ENT-A-CTR-01", "ENT-A-ZZZ-99"))
    ),
    "N6_weak_catch_all_runtime_prefix": lambda t: once(
        t,
        "PDC_DOMAIN_IMPACT_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (\n",
        'PDC_DOMAIN_IMPACT_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (\n    ("intergrax/runtime", ("ENT-A-DOC-01",)),\n',
    ),
    "N7_tier3_root_narrowed_to_one_mcp_app": lambda t: edit_tier3(
        t, lambda b: once(b, '"applications",', '"applications/lab_application/mcp",')
    ),
    "N8_reroute_existing_tier0_applications_rule": lambda t: once(
        t, '"intergrax/applications",', '"intergrax/applications_x",'
    ),
}


def pytest_run() -> subprocess.CompletedProcess[str]:
    return run([sys.executable, "-m", "pytest", *TESTS, "-q", "--no-header", "-p", "no:randomly", "-rf"])


results: dict[str, object] = {}
restore()
baseline = pytest_run()
control_failed = set(re.findall(r"^FAILED (\S+)", baseline.stdout, flags=re.M))
results["CONTROL_unmutated"] = {
    "exit": baseline.returncode,
    "failed_tests": sorted(control_failed),
    "tail": baseline.stdout.strip().splitlines()[-3:],
}
for name, mutate in MUTANTS.items():
    restore()
    try:
        IMPACT.write_text(mutate(IMPACT.read_text(encoding="utf-8")), encoding="utf-8")
        gen = run([sys.executable, "scripts/enterprise_audit/build_impact_map.py", "--emit-ssot"])
        out = pytest_run()
        failed = re.findall(r"^FAILED (\S+)", out.stdout, flags=re.M)
        new = sorted(set(failed) - control_failed)
        results[name] = {
            "generator_exit": gen.returncode,
            "pytest_exit": out.returncode,
            "verdict": "KILLED" if new else "SURVIVED",
            "new_failures_vs_control": new,
            "tail": out.stdout.strip().splitlines()[-2:],
        }
    except Exception as exc:  # noqa: BLE001
        results[name] = {"error": f"{type(exc).__name__}: {exc}"}
    finally:
        restore()
print(json.dumps(results, indent=2))
