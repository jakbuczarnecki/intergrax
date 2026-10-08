# © Artur Czarnecki. All rights reserved.

"""TRACE-X-P5-R2-P3-R1 mechanical production host composition gates."""

from __future__ import annotations

import ast
import importlib
import inspect
import subprocess
from pathlib import Path

import pytest

pytestmark = [pytest.mark.qualification, pytest.mark.gate]

_TRACE_X_P5_R2_P3_R1_START_HEAD = "6519a8263edd92a1cc600a0ce1d0e20e57829e84"
_REPO_ROOT = Path(__file__).resolve().parents[3]
_COMPOSITION_MODULE = (
    _REPO_ROOT
    / "intergrax/applications/_shared/uca6c_marketplace_qualified_execution_composition.py"
)


def test_txp5r2p3r1_q01_start_head_ancestry() -> None:
    subprocess.check_call(
        ["git", "merge-base", "--is-ancestor", _TRACE_X_P5_R2_P3_R1_START_HEAD, "HEAD"],
        cwd=_REPO_ROOT,
    )


def test_txp5r2p3r1_q02_production_composition_module_exists() -> None:
    mod = importlib.import_module(
        "intergrax.applications._shared.uca6c_marketplace_qualified_execution_composition",
    )
    assert hasattr(mod, "build_production_marketplace_qualified_capability_execution_handler")


def test_txp5r2p3r1_q03_uses_sanctioned_pinning_wire() -> None:
    source = _COMPOSITION_MODULE.read_text(encoding="utf-8")
    assert "wire_execution_integration_configuration_pinning_store" in source
    assert "InMemoryExecutionIntegrationConfigurationPinningStore" not in source


def test_txp5r2p3r1_q04_builds_pattern_a_resolution() -> None:
    source = _COMPOSITION_MODULE.read_text(encoding="utf-8")
    assert "ExecutionBoundIntegrationResolution" in source
    assert source.count("ExecutionBoundIntegrationResolution(") == 1


def test_txp5r2p3r1_q05_canonical_relational_binding_builder() -> None:
    source = _COMPOSITION_MODULE.read_text(encoding="utf-8")
    assert "build_default_configured_relational_store_execution_binding" in source
    assert "integration_profile=" not in source


def test_txp5r2p3r1_q06_default_configured_projection() -> None:
    source = _COMPOSITION_MODULE.read_text(encoding="utf-8")
    assert "DefaultConfiguredIntegrationToolInvocationProjectionPort" in source


def test_txp5r2p3r1_q07_marketplace_handler_receives_projection() -> None:
    source = _COMPOSITION_MODULE.read_text(encoding="utf-8")
    assert "configured_invocation_projection=projection" in source


def test_txp5r2p3r1_q08_no_sqlite_in_composition_module() -> None:
    source = _COMPOSITION_MODULE.read_text(encoding="utf-8").lower()
    assert "sqlite" not in source


def test_txp5r2p3r1_q09_aw_composition_has_no_pinning_store() -> None:
    source = (
        _REPO_ROOT
        / "intergrax/autonomous_work/worker_recovery_governed_fulfillment_composition.py"
    ).read_text(encoding="utf-8")
    assert "wire_execution_integration_configuration_pinning_store" not in source
    assert "ConfiguredIntegrationToolInvocationProjectionPort" not in source


def test_txp5r2p3r1_q10_dispatch_helper_registers_single_handler() -> None:
    from intergrax.applications._shared.uca6c_marketplace_qualified_execution_composition import (
        build_production_marketplace_qualified_capability_execution_dispatch,
    )

    assert "QualifiedCapabilityExecutionBindingHandlerRegistry" in inspect.getsource(
        build_production_marketplace_qualified_capability_execution_dispatch,
    )


def test_txp5r2p3r1_q11_production_builder_fail_fast_backing() -> None:
    from intergrax.applications._shared.uca6c_marketplace_qualified_execution_composition import (
        Uca6cMarketplaceQualifiedExecutionCompositionError,
        _require_exactly_one_pinning_backing,
    )

    with pytest.raises(Uca6cMarketplaceQualifiedExecutionCompositionError):
        _require_exactly_one_pinning_backing(
            configuration_pinning_kv_store=None,
            configuration_pinning_document_store=None,
        )


def test_txp5r2p3r1_q12_no_getattr_durability_probing() -> None:
    tree = ast.parse(_COMPOSITION_MODULE.read_text(encoding="utf-8"))
    source = ast.unparse(tree)
    assert "getattr(" not in source
    assert "hasattr(" not in source
