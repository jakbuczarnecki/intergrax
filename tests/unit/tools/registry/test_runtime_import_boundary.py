# © Artur Czarnecki. All rights reserved.

"""UCA-6C-R6-CERT — ToolRegistry runtime leaf import must not load composition graph."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[4]

_FORBIDDEN_PREFIXES = (
    "intergrax.tools.registry.wiring",
    "intergrax.tools.registry.catalog",
    "intergrax.tools.registry.bootstrap",
    "intergrax.tools.registry.factory",
    "intergrax.integrations.registry",
    "intergrax.integrations.providers.relational_store.sqlite",
    "intergrax.collaborative_work",
    "intergrax.runtime.integrations.categories",
)


def _run_import_subprocess(statement: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", statement],
        cwd=_REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def test_tool_registry_runtime_cold_import_in_subprocess() -> None:
    forbidden_literal = repr(_FORBIDDEN_PREFIXES)
    statement = (
        "import sys\n"
        "from intergrax.tools.registry.runtime import ToolRegistry\n"
        f"forbidden = {forbidden_literal}\n"
        "loaded = [name for name in sys.modules if any(name.startswith(p) for p in forbidden)]\n"
        "assert not loaded, f'unexpected modules loaded: {loaded}'\n"
        "assert ToolRegistry is not None\n"
    )
    completed = _run_import_subprocess(statement)
    assert completed.returncode == 0, completed.stdout + completed.stderr
