# © Artur Czarnecki. All rights reserved.

"""TRACE-X-P5-R2-P4 reconstruction projection qualification gates."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from tests.qualification.trace_x._trace_x_p5_r2_p4_support import (
    FORBIDDEN_PROJECTION_IMPORT_PREFIXES,
    FORBIDDEN_RECONSTRUCTOR_IMPORT_PREFIXES,
    PROJECTION_MODULE,
    READER_MODULE,
    RECONSTRUCTOR_MODULE,
    TRACE_X_P5_R2_P4_START_HEAD,
    count_class_definitions,
    module_import_prefix_violations,
)

pytestmark = [pytest.mark.qualification, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_P4_UNIT = (
    _REPO_ROOT
    / "tests/unit/runtime/observability/reconstruction/test_trace_x_p5_r2_p4_integration_configuration_provenance.py"
)


def test_txp5r2p4_q01_start_head_ancestry() -> None:
    subprocess.check_call(
        ["git", "merge-base", "--is-ancestor", TRACE_X_P5_R2_P4_START_HEAD, "HEAD"],
    )


def test_txp5r2p4_q02_single_execution_reconstructor() -> None:
    assert count_class_definitions(RECONSTRUCTOR_MODULE, "ExecutionReconstructor") == 1


def test_txp5r2p4_q03_reconstructor_neutral_reader_only() -> None:
    violations = module_import_prefix_violations(
        RECONSTRUCTOR_MODULE,
        FORBIDDEN_RECONSTRUCTOR_IMPORT_PREFIXES,
    )
    assert not violations, violations
    source = RECONSTRUCTOR_MODULE.read_text(encoding="utf-8")
    assert "ExecutionIntegrationConfigurationProvenanceReader" in source
    assert "ExecutionIntegrationConfigurationPinningStore" not in source


def test_txp5r2p4_q04_projection_has_no_integrations_implementation_imports() -> None:
    violations = module_import_prefix_violations(
        PROJECTION_MODULE,
        FORBIDDEN_PROJECTION_IMPORT_PREFIXES,
    )
    assert not violations, violations


def test_txp5r2p4_q05_reader_adapter_present() -> None:
    source = READER_MODULE.read_text(encoding="utf-8")
    assert "PinningStoreExecutionIntegrationConfigurationProvenanceReader" in source
    assert "read_all" in source
    assert "pin(" not in source.split("class PinningStore")[1]


def test_txp5r2p4_q06_diagnostic_composition_injects_reader() -> None:
    path = _REPO_ROOT / "intergrax/applications/_shared/diagnostic_composition.py"
    source = path.read_text(encoding="utf-8")
    assert "execution_integration_configuration_provenance_reader" in source


def test_txp5r2p4_q07_harness_wires_shared_pinning_reader() -> None:
    path = _REPO_ROOT / "intergrax/applications/_shared/harness_host_runtime.py"
    source = path.read_text(encoding="utf-8")
    assert "resolve_pinning_store_integration_configuration_provenance_reader" in source


def test_txp5r2p4_q08_p4_unit_regression_tests_present() -> None:
    source = _P4_UNIT.read_text(encoding="utf-8")
    for name in (
        "test_historical_restart_ignores_changed_current_configuration_state",
        "test_child_execution_does_not_inherit_parent_provenance",
        "test_required_provenance_missing_fails_closed",
    ):
        assert name in source
