"""P7-00 addendum 3: mutation sample on the ollama dispatch wrapper and the ebh_5 harness precondition (auditor-owned).

usage: python addendum_3_mutation_sample.py <scratch-checkout>
For each mutant: edit register.py (or the harness primitives), run the targeted tests once, restore the file with git.
The control run (unmutated) is recorded first; a mutant is KILLED only when it makes a test fail that was not red in the
control run. One pytest process at a time.
Targeted tests: the new p6_ecapfix and the B-REG-P reg_p tests, the embedding-provider catalog unit tests, the registry
unit tests and the two ebh_5 package C v2 embedding cases (-k embedding_provider).
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
REG = "intergrax/integrations/providers/embedding_provider/ollama/register.py"
PRIM = "tests/qualification/ebh_5/_package_c_v2_primitives.py"
TESTS = [
    "tests/qualification/pdc_x/p6_ecapfix",
    "tests/qualification/pdc_x/reg_p",
    "tests/unit/integrations/test_p2_002_b2_embedding_provider_catalog.py",
    "tests/unit/integrations/test_p2_003_explicit_contract_specs.py",
    "tests/unit/integrations/registry",
]
EBH = ["tests/qualification/ebh_5/test_ebh_5_behavioral_package_c_v2.py", "-k", "embedding_provider"]
env = dict(os.environ)
env["PYTHONPATH"] = str(root).replace("\\", "/") + os.pathsep + env.get("PYTHONPATH", "")
env["PYTHONUTF8"] = "1"
env["PYTHONDONTWRITEBYTECODE"] = "1"


def run(cmd: list[str], timeout: int = 900) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=root, env=env, capture_output=True, text=True, timeout=timeout)


def restore() -> None:
    run(["git", "checkout", "--", REG, PRIM])


def sub_once(text: str, old: str, new: str) -> str:
    assert text.count(old) == 1, f"mutation anchor not unique or missing: {old!r}"
    return text.replace(old, new)


MUTANTS: dict[str, tuple[str, object]] = {
    "M1_invert_dispatch_condition": (REG, lambda t: sub_once(t, '== IntegrationCategory.EMBEDDING_PROVIDER', '!= IntegrationCategory.EMBEDDING_PROVIDER')),
    "M2_always_owning_factory_revert_fix": (REG, lambda t: sub_once(t, 'if kwargs.get("integration_category") == IntegrationCategory.EMBEDDING_PROVIDER:', 'if False:')),
    "M3_always_embedding_branch": (REG, lambda t: sub_once(t, 'if kwargs.get("integration_category") == IntegrationCategory.EMBEDDING_PROVIDER:', 'if True:')),
    "M4_embedding_enabled_hard_true": (REG, lambda t: sub_once(t, 'enabled=bool(kwargs.get("enabled", False))', 'enabled=True')),
    "M5_embedding_enabled_default_true": (REG, lambda t: sub_once(t, 'bool(kwargs.get("enabled", False))', 'bool(kwargs.get("enabled", True))')),
    "M6_embedding_ignores_enabled": (REG, lambda t: sub_once(t, 'enabled=bool(kwargs.get("enabled", False))', 'enabled=False')),
    "M7_drop_wrapper_registration": (REG, lambda t: sub_once(t, '    register_integration(replace(entry, factory=_dispatching_factory(entry.factory)), override=True)', '    pass')),
    "M8_wrapper_registration_without_override": (REG, lambda t: sub_once(t, 'factory=_dispatching_factory(entry.factory)), override=True)', 'factory=_dispatching_factory(entry.factory)), override=False)')),
    "M9_owning_factory_called_without_kwargs": (REG, lambda t: sub_once(t, 'return owning_factory(**kwargs)', 'return owning_factory()')),
    "M10_dispatch_also_routes_model_serving": (REG, lambda t: sub_once(t, 'if kwargs.get("integration_category") == IntegrationCategory.EMBEDDING_PROVIDER:', 'if kwargs.get("integration_category") in (IntegrationCategory.EMBEDDING_PROVIDER, IntegrationCategory.MODEL_SERVING_RUNTIME):')),
    "M11_equivalent_str_comparison": (REG, lambda t: sub_once(t, 'kwargs.get("integration_category") == IntegrationCategory.EMBEDDING_PROVIDER', 'str(getattr(kwargs.get("integration_category"), "value", kwargs.get("integration_category"))) == "embedding_provider"')),
    "M12_harness_drops_owning_row_registration": (PRIM, lambda t: sub_once(t, '        register_ollama_integration()\n', '        pass\n')),
}

results: dict[str, object] = {}


def pytest_run() -> subprocess.CompletedProcess[str]:
    """Two processes in sequence (the -k filter of the EBH file must not deselect the other tests); outputs are joined."""
    a = run([sys.executable, "-m", "pytest", *TESTS, "-q", "--no-header", "-p", "no:randomly", "-rf"])
    b = run([sys.executable, "-m", "pytest", *EBH, "-q", "--no-header", "-p", "no:randomly", "-rf"])
    return subprocess.CompletedProcess(a.args, max(a.returncode, b.returncode), a.stdout + chr(10) + b.stdout, "")


restore()
baseline = pytest_run()
control_failed = set(re.findall(r"^FAILED (\S+)", baseline.stdout, flags=re.M))
results["CONTROL_unmutated"] = {
    "exit": baseline.returncode,
    "failed_tests": sorted(control_failed),
    "tail": [l for l in baseline.stdout.splitlines() if " passed" in l or " failed" in l][-3:],
}
for name, (rel, mutate) in MUTANTS.items():
    restore()
    target = root / rel
    try:
        target.write_text(mutate(target.read_text(encoding="utf-8")), encoding="utf-8")  # type: ignore[operator]
        out = pytest_run()
        failed = re.findall(r"^FAILED (\S+)", out.stdout, flags=re.M)
        new = sorted(set(failed) - control_failed)
        results[name] = {
            "file": rel,
            "pytest_exit": out.returncode,
            "verdict": "KILLED" if new else "SURVIVED",
            "new_failures_vs_control": new,
            "tail": [l for l in out.stdout.splitlines() if " passed" in l or " failed" in l][-2:],
        }
    except Exception as exc:  # noqa: BLE001
        results[name] = {"error": f"{type(exc).__name__}: {exc}"}
    finally:
        restore()
print(json.dumps(results, indent=2))
