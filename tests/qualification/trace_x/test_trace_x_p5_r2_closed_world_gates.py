# © Artur Czarnecki. All rights reserved.

"""TRACE-X-P5-R2 closed-world mechanical gates for configured/effective provenance."""

from __future__ import annotations

import ast
import subprocess
from pathlib import Path

import pytest

from tests.qualification.trace_x._trace_x_p5_r2_closed_world_adversarial_matrix import (
    P5_SEMANTIC_OWNER_MATRIX,
)
from tests.qualification.trace_x._trace_x_p5_r2_closed_world_support import (
    TRACE_X_P5_R2_CLOSED_WORLD_START_HEAD,
    compare_paths_to_registry,
    discover_configured_execution_path_keys,
    grep_production_pattern,
)
from tests.qualification.trace_x._trace_x_p5_r2_configured_execution_path_registry import (
    CONFIGURED_EXECUTION_PATH_REGISTRY,
)

pytestmark = [pytest.mark.qualification, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_INTERGRAX = _REPO_ROOT / "intergrax"
_COMPOSITION = (
    _INTERGRAX / "applications/_shared/uca6c_marketplace_qualified_execution_composition.py"
)
_PROJECTION = (
    _INTERGRAX
    / "runtime/observability/reconstruction/integration_configuration_provenance_projection.py"
)
_RECONSTRUCTOR = _INTERGRAX / "runtime/observability/reconstruction/execution_reconstruction.py"
_RESOLUTION = _INTERGRAX / "integrations/execution_bound_integration_resolution.py"
_RECORDER = (
    _INTERGRAX / "runtime/execution/integration_configuration_provenance_requirement_recorder.py"
)
_BINDING = _INTERGRAX / "integrations/configured_relational_store_execution_binding.py"


def test_txp5cw_q01_start_head_ancestry() -> None:
    subprocess.check_call(
        ["git", "merge-base", "--is-ancestor", TRACE_X_P5_R2_CLOSED_WORLD_START_HEAD, "HEAD"],
        cwd=_REPO_ROOT,
    )


def test_txp5cw_q02_start_head_object_exists() -> None:
    subprocess.check_call(
        ["git", "cat-file", "-e", f"{TRACE_X_P5_R2_CLOSED_WORLD_START_HEAD}^{{commit}}"],
        cwd=_REPO_ROOT,
    )


def test_txp5cw_q03_configured_execution_paths_closed_world_parity() -> None:
    discovered = discover_configured_execution_path_keys()
    result = compare_paths_to_registry(discovered, CONFIGURED_EXECUTION_PATH_REGISTRY)
    assert result.duplicate_registry_keys == frozenset()
    assert not result.unknown, f"unclassified discovery: {sorted(result.unknown)}"
    assert not result.orphan, f"orphan registry rows: {sorted(result.orphan)}"
    assert not result.production_bypass, f"bypass: {sorted(result.production_bypass)}"
    assert not result.unclassified, f"F-class rows: {sorted(result.unclassified)}"
    assert result.ok


def test_txp5cw_q04_negative_sensitivity_unregistered_module() -> None:
    baseline = compare_paths_to_registry(
        discover_configured_execution_path_keys(),
        CONFIGURED_EXECUTION_PATH_REGISTRY,
    )
    assert baseline.ok
    synthetic = ("intergrax/runtime/example_new_configured_path.py", "module")
    failed = compare_paths_to_registry(
        discover_configured_execution_path_keys() | {synthetic},
        CONFIGURED_EXECUTION_PATH_REGISTRY,
    )
    assert synthetic in failed.unknown
    assert not failed.ok


def test_txp5cw_q05_production_composition_requires_requirement_evidence() -> None:
    source = _COMPOSITION.read_text(encoding="utf-8")
    assert "require_configured_adopted_requirement_evidence=True" in source


def test_txp5cw_q06_single_execution_bound_integration_resolution_class() -> None:
    tree = ast.parse(_RESOLUTION.read_text(encoding="utf-8"))
    classes = [
        node.name
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "ExecutionBoundIntegrationResolution"
    ]
    assert len(classes) == 1


def test_txp5cw_q07_no_second_configured_resolver_class_names() -> None:
    forbidden = (
        "ConfiguredIntegrationResolver",
        "ConfiguredProviderResolver",
        "ExecutionConfiguredProviderResolver",
    )
    for name in forbidden:
        hits = grep_production_pattern(rf"class {name}\b")
        assert hits == [], f"duplicate resolver {name} in {hits}"


def test_txp5cw_q08_single_requirement_recorder_emitter() -> None:
    tree = ast.parse(_RECORDER.read_text(encoding="utf-8"))
    classes = [
        node.name
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and "Requirement" in node.name
        and "Recorder" in node.name
    ]
    assert len(classes) == 1
    recorder_source = _RECORDER.read_text(encoding="utf-8")
    assert "INTEGRATION_CONFIGURATION_PROVENANCE_REQUIREMENT_COMMITTED" in recorder_source


def test_txp5cw_q09_magic_requirement_flag_authority_zero_in_production() -> None:
    assert (
        grep_production_pattern(r"execution_integration_configuration_provenance_required")
        == []
    )


def test_txp5cw_q10_reconstruction_no_current_configuration_opportunity_reads() -> None:
    for path in (_PROJECTION, _RECONSTRUCTOR):
        text = path.read_text(encoding="utf-8").lower()
        assert "opportunity_read" not in text
        assert "resolve_from_profile" not in text
        assert "existing_capability_configuration" not in text


def test_txp5cw_q11_production_direct_port_construction_count() -> None:
    """ExecutionBoundConfiguredRelationalStorePort must be constructed only in canonical factory."""
    allowed = {
        "intergrax/integrations/configured_relational_store_execution_binding.py",
        "intergrax/integrations/execution_bound_configured_relational_store_port.py",
    }
    hits = grep_production_pattern(r"ExecutionBoundConfiguredRelationalStorePort\s*\(")
    assert set(hits) <= allowed, f"unexpected direct port construction: {set(hits) - allowed}"


def test_txp5cw_q12_external_work_not_in_configured_adopted_registry_surfaces() -> None:
    for row in CONFIGURED_EXECUTION_PATH_REGISTRY:
        assert "external_work" not in row.path.lower()


def test_txp5cw_q13_semantic_owner_duplicate_matrix() -> None:
    for concern, _owner, count in P5_SEMANTIC_OWNER_MATRIX:
        assert concern
        assert count == 1


def test_txp5cw_q14_binding_default_requires_evidence_flag_surface() -> None:
    source = _BINDING.read_text(encoding="utf-8")
    assert "require_configured_adopted_requirement_evidence" in source


def test_txp5cw_q15_diagnostics_reader_has_no_pin_or_adoption_mutations() -> None:
    reader = (
        _INTERGRAX / "applications/_shared/integrations/integration_configuration_provenance_reader.py"
    ).read_text(encoding="utf-8")
    assert ".pin(" not in reader
    assert "ExecutionIntegrationConfigurationAdoption(" not in reader
