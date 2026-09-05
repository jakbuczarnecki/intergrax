# © Artur Czarnecki. All rights reserved.

"""P2-003-D2-V2: contracts package import boundary regression gate."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[3]

_FORBIDDEN_RUNTIME_MODULES = (
    "intergrax.runtime.nexus",
    "intergrax.runtime.diagnostics",
    "intergrax.runtime.observability",
    "intergrax.contracts.runtime_mapping",
)

_LEAF_CONTRACT_IMPORT = """
import sys
from intergrax.contracts.execution_phase import ExecutionPhase
assert ExecutionPhase is not None
forbidden = {forbidden!r}
for name in forbidden:
    assert name not in sys.modules, f"forbidden module loaded: {{name}}"
print("leaf contract import OK")
""".format(forbidden=_FORBIDDEN_RUNTIME_MODULES)

_FACADE_IMPORTS_WITHOUT_RUNTIME_MAPPING = """
import sys
from intergrax.contracts import AgentContract, ExecutionInterrupt
assert AgentContract is not None
assert ExecutionInterrupt is not None
assert "intergrax.contracts.runtime_mapping" not in sys.modules
print("facade imports OK")
"""

_LAZY_RUNTIME_MAPPING = """
import sys
import intergrax.contracts as contracts
assert "intergrax.contracts.runtime_mapping" not in sys.modules
fn = contracts.runtime_answer_to_agent_result
assert callable(fn)
assert "intergrax.contracts.runtime_mapping" in sys.modules
print("lazy runtime mapping OK")
"""

_PUBLIC_CONTRACT_IMPORTS = (
    "from intergrax.contracts.execution_phase import ExecutionPhase; assert ExecutionPhase is not None",
    "from intergrax.contracts import AgentContract; assert AgentContract is not None",
    "from intergrax.contracts import ExecutionInterrupt; assert ExecutionInterrupt is not None",
    "from intergrax.contracts import runtime_answer_to_agent_result; assert callable(runtime_answer_to_agent_result)",
)


def _run_import_subprocess(statement: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", statement],
        cwd=_REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def test_leaf_contract_import_does_not_initialize_runtime() -> None:
    completed = _run_import_subprocess(_LEAF_CONTRACT_IMPORT)
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_contracts_facade_avoids_runtime_mapping_until_accessed() -> None:
    completed = _run_import_subprocess(_FACADE_IMPORTS_WITHOUT_RUNTIME_MAPPING)
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_runtime_answer_to_agent_result_lazy_facade() -> None:
    completed = _run_import_subprocess(_LAZY_RUNTIME_MAPPING)
    assert completed.returncode == 0, completed.stdout + completed.stderr


@pytest.mark.parametrize("statement", _PUBLIC_CONTRACT_IMPORTS)
def test_public_contract_imports(statement: str) -> None:
    completed = _run_import_subprocess(statement)
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_contracts_package_init_does_not_eagerly_import_runtime_mapping() -> None:
    init_path = _REPO_ROOT / "intergrax" / "contracts" / "__init__.py"
    source = init_path.read_text(encoding="utf-8")
    forbidden_prefixes = ("intergrax.runtime", "intergrax.decision")
    skip_regions = ("if TYPE_CHECKING:", "def __getattr__")
    skipping = False
    for line in source.splitlines():
        stripped = line.strip()
        if any(marker in stripped for marker in skip_regions):
            skipping = True
            continue
        if skipping:
            if stripped and not line.startswith((" ", "\t")):
                skipping = False
            else:
                continue
        if not stripped.startswith("from ") and not stripped.startswith("import "):
            continue
        for prefix in forbidden_prefixes:
            assert prefix not in stripped, (
                f"contracts/__init__.py imports forbidden dependency: {line}"
            )
        assert "runtime_mapping" not in stripped, (
            f"contracts/__init__.py must lazy-load runtime_mapping: {line}"
        )
