"""P7-00 addendum 2: mutation sample on the impact map (auditor-owned, runs ONLY in a scratch detached checkout).

usage: python addendum_2_mutation_sample.py <scratch-checkout>
For each mutant: edit impact_map.py, regenerate the SSOT JSON with scripts/enterprise_audit/build_impact_map.py --emit-ssot
(so the shipped-JSON-equals-generator test is not the only killer), run the targeted tests once, restore the files with git.
The targeted tests are the ones of the T-IGX-036-V test contract that concern the map: pdc_x/p6_impact, xd_12 and the
ent_audit_x unit test of the impact map. One pytest process at a time.
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
env["PYTHONPATH"] = str(root) + os.pathsep + env.get("PYTHONPATH", "")
env["PYTHONUTF8"] = "1"


def run(cmd: list[str], timeout: int = 900) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=root, env=env, capture_output=True, text=True, timeout=timeout)


def restore() -> None:
    run(["git", "checkout", "--", str(IMPACT.relative_to(root)), str(SSOT.relative_to(root))])


def sub_once(text: str, pattern: str, repl: str) -> str:
    new, n = re.subn(pattern, repl, text, count=1, flags=re.S)
    assert n == 1, f"mutation pattern not found: {pattern}"
    return new


def rule_block(prefix: str) -> str:
    # a rule entry "(\n\"prefix\", (...),\n)," or inline ("prefix", (...)),
    return r'(\(\s*"' + re.escape(prefix) + r'",\s*\()[^)]*(\),?\s*\),)'


MUTANTS = {
    "M1_drop_security_rule": lambda t: sub_once(t, r'    # immutable security audit trail script \+ report redaction test.*?\n    \(\n        "intergrax/runtime/security",.*?\n    \),\n', ""),
    "M2_reroute_security_to_doc_check": lambda t: sub_once(t, rule_block("intergrax/runtime/security"), r'\1"ENT-A-DOC-01",\2'),
    "M3_reroute_agent_governance_to_doc_check": lambda t: sub_once(t, r'("intergrax/runtime/agent_governance", \()[^)]*(\)\),)', r'\1"ENT-A-DOC-01",\2'),
    "M4_shrink_applications_to_one_id": lambda t: sub_once(t, rule_block("intergrax/applications"), r'\1"ENT-A-BND-01",\2'),
    "M5_own_files_back_to_legacy_ids": lambda t: sub_once(t, r'\("intergrax/platform/enterprise_audit/impact_map.py", \("ENT-A-EVOL-01", "ENT-A-EVOL-02", "ENT-A-EVOL-03"\)\)', '("intergrax/platform/enterprise_audit/impact_map.py", ("C024", "C029", "C072"))'),
    "M6_unknown_id_in_domain_rule": lambda t: sub_once(t, r'("intergrax/skills", \()', r'\1"ENT-A-ZZZ-99", '),
    "M7_weak_catch_all_runtime_prefix": lambda t: sub_once(t, r"PDC_DOMAIN_IMPACT_RULES: tuple\[tuple\[str, tuple\[str, \.\.\.\]\], \.\.\.\] = \(\n", 'PDC_DOMAIN_IMPACT_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (\n    ("intergrax/runtime", ("ENT-A-DOC-01",)),\n'),
}

results = {}
baseline = run([sys.executable, "-m", "pytest", *TESTS, "-q", "--no-header", "-p", "no:randomly"])
control_failed = set(re.findall(r"^FAILED (\S+)", baseline.stdout, flags=re.M))
results["CONTROL_unmutated"] = {"exit": baseline.returncode, "failed_tests": sorted(control_failed), "tail": baseline.stdout.strip().splitlines()[-3:]}
for name, mutate in MUTANTS.items():
    restore()
    try:
        IMPACT.write_text(mutate(IMPACT.read_text(encoding="utf-8")), encoding="utf-8")
        gen = run([sys.executable, "scripts/enterprise_audit/build_impact_map.py", "--emit-ssot"])
        out = run([sys.executable, "-m", "pytest", *TESTS, "-q", "--no-header", "-p", "no:randomly"])
        failed = re.findall(r"^FAILED (\S+)", out.stdout, flags=re.M)
        results[name] = {
            "generator_exit": gen.returncode,
            "pytest_exit": out.returncode,
            "verdict": "KILLED" if (set(failed) - control_failed) else "SURVIVED",
            "new_failures_vs_control": sorted(set(failed) - control_failed),
            "failed_tests": failed,
            "tail": out.stdout.strip().splitlines()[-2:],
        }
    except Exception as exc:  # noqa: BLE001
        results[name] = {"error": f"{type(exc).__name__}: {exc}"}
    finally:
        restore()
print(json.dumps(results, indent=2))
