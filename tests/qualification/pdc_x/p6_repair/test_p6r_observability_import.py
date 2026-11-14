# © Artur Czarnecki. All rights reserved.

"""B-P6-R item 5 - no circular ImportError when the integration registry is the first import."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[4]

_FIRST_IMPORTS = [
    pytest.param(
        "from intergrax.integrations.registry.bootstrap import register_default_integrations",
        id="registry-first",
    ),
    pytest.param(
        "from intergrax.integrations.registry.bootstrap import register_default_integrations; "
        "register_default_integrations(preset='core')",
        id="registry-first-core-preset",
    ),
    pytest.param(
        "from intergrax.integrations.registry.bootstrap import register_default_integrations; "
        "register_default_integrations(preset='full')",
        id="registry-first-full-preset",
    ),
    pytest.param(
        "from intergrax.runtime.integrations.observability import ObservabilityVendorIntegrationContract",
        id="observability-first",
    ),
    pytest.param(
        "from intergrax.runtime.integrations.categories import ObservabilityVendorIntegrationContract",
        id="categories-first",
    ),
    pytest.param(
        "from intergrax.runtime.observability.export_boundary import ObservabilityExportEnvelope",
        id="export-boundary-first",
    ),
]


@pytest.mark.parametrize("statement", _FIRST_IMPORTS)
def test_the_statement_succeeds_as_the_first_import_of_a_fresh_interpreter(statement: str) -> None:
    result = subprocess.run(
        [sys.executable, "-c", statement],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stderr[-1500:]
    assert "circular import" not in result.stderr


def test_the_observability_contract_keeps_its_public_surface() -> None:
    from intergrax.runtime.integrations import observability

    for name in (
        "ObservabilityVendorIntegrationContract",
        "ObservabilityVendorPayload",
        "map_envelope_to_vendor_payload",
        "require_policy_sanitized_envelope",
    ):
        assert hasattr(observability, name), name
