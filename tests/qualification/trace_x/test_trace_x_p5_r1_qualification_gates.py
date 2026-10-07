# © Artur Czarnecki. All rights reserved.

"""TRACE-X-P5-R1 mechanical qualification gates."""

from __future__ import annotations

import ast
import inspect
import subprocess
from pathlib import Path

import pytest

from intergrax.runtime.execution.environment_host_task_execution import (
    build_environment_host_task_execution,
)
from tests.qualification.trace_x._trace_x_p5_r1_child_discovery import (
    discover_canonical_constructor_rebinding_violations,
    discover_canonical_constructor_rebindings_in_source,
    discover_child_execution_runner_constructor_surfaces,
    discover_child_execution_runner_surfaces_in_source,
    discover_wire_host_effective_profile_execution_roots,
    discover_wire_host_roots_in_source,
    profile_aware_root_forwards_child_context_inheritance,
)
from tests.qualification.trace_x._trace_x_p5_r1_child_registry import (
    CHILD_EXECUTION_RUNNER_SURFACE_REGISTRY,
    GENERIC_COMPOSITION_DRIFT_WATCH_REGISTRY,
    PROFILE_AWARE_WIRE_HOST_ROOT_REGISTRY,
    compare_child_runner_surfaces_to_registry,
    compare_profile_aware_wire_host_roots_to_registry,
    registry_classification_ambiguity_violations,
)
from tests.qualification.trace_x._trace_x_p5_r1_resume_evidence import (
    RESUME_BASELINE_SHA,
    RESUME_EVIDENCE_SCHEMA_VERSION,
    RESUME_TEST_NODE_IDS,
    derive_resume_comparison,
    load_resume_baseline_evidence,
    validate_resume_baseline_evidence,
    validate_resume_baseline_payload,
)
from tests.qualification.trace_x._trace_x_p5_r1_support import (
    TRACE_X_P5_R1_R1_Q1_START_HEAD,
    TRACE_X_P5_R1_R1_Q2_START_HEAD,
    TRACE_X_P5_R1_R1_Q3_START_HEAD,
    TRACE_X_P5_R1_R1_Q4_START_HEAD,
    TRACE_X_P5_R1_START_HEAD,
    child_runner_profile_resolution_import_violations,
    discover_profile_aware_environment_host_roots,
    global_profile_child_inheritance_structural_chain_gaps,
    neutral_revision_ref_shadow_grammar_violations,
    production_child_identity_mint_surface_violations,
    profile_aware_orchestration_spec_wiring_gaps,
    profile_child_inheritance_has_production_caller,
    reconstruction_profile_fallback_sentinels,
    reconstruction_runtime_import_violations,
)
from tests.qualification.trace_x._trace_x_p4_registry_types import compare_discovered_to_registry

pytestmark = [pytest.mark.qualification, pytest.mark.gate]


def test_txp5r1_q01_start_head_ancestry() -> None:
    subprocess.check_call(
        ["git", "merge-base", "--is-ancestor", TRACE_X_P5_R1_START_HEAD, "HEAD"],
    )


def test_txp5r1_q02_reconstruction_no_application_profile_imports() -> None:
    violations = reconstruction_runtime_import_violations()
    assert not violations, f"forbidden reconstruction imports: {violations}"


def test_txp5r1_q03_reconstruction_no_profile_fallback_sentinels() -> None:
    hits = reconstruction_profile_fallback_sentinels()
    assert not hits, f"forbidden profile fallback patterns: {hits}"


def test_txp5r1_q04_environment_host_revision_admission_mandatory() -> None:
    signature = inspect.signature(build_environment_host_task_execution)
    param = signature.parameters["revision_admission"]
    assert param.default is inspect.Parameter.empty


def test_txp5r1_q05_profile_aware_roots_wire_revision_admission() -> None:
    missing = discover_profile_aware_environment_host_roots()
    assert not missing, f"profile-aware roots missing revision_admission: {missing}"


def test_txp5r1_q06_neutral_revision_ref_no_shadow_grammar() -> None:
    violations = neutral_revision_ref_shadow_grammar_violations()
    assert not violations, violations


def test_txp5r1_q07_child_runner_modules_no_profile_resolution_imports() -> None:
    violations = child_runner_profile_resolution_import_violations()
    assert not violations, violations


def test_txp5r1_q08_inherit_child_execution_pinned_revision_production_caller() -> None:
    assert profile_child_inheritance_has_production_caller()


def test_txp5r1_q09_profile_aware_orchestration_child_context_wiring() -> None:
    missing = profile_aware_orchestration_spec_wiring_gaps()
    assert not missing, missing


def test_txp5r1_q10_graph_executor_accepts_child_context_inheritance() -> None:
    from intergrax.runtime.nexus.execution.graph_executor import GraphExecutor

    assert "child_context_inheritance" in inspect.signature(GraphExecutor.__init__).parameters


def test_txp5r1_q11_policy_canonical_chain_test_present() -> None:
    path = (
        Path(__file__).resolve().parents[2]
        / "unit"
        / "runtime"
        / "observability"
        / "reconstruction"
        / "test_p5_r1_policy_profile_provenance.py"
    )
    text = path.read_text(encoding="utf-8")
    assert "test_policy_provenance_from_canonical_governance_persistence_chain" in text
    assert "payload_schema_id" in text


def test_txp5r1_q12_child_runner_discovery_registry_parity() -> None:
    discovered = discover_child_execution_runner_constructor_surfaces()
    result = compare_child_runner_surfaces_to_registry(discovered)
    assert not result.unknown, f"unclassified child runner surfaces: {sorted(result.unknown)}"
    assert not result.orphan, f"orphan child runner registry rows: {sorted(result.orphan)}"
    assert not result.duplicate_registry_keys
    assert len(discovered) == len(CHILD_EXECUTION_RUNNER_SURFACE_REGISTRY)


def test_txp5r1_q13_child_runner_registry_classification_explicit() -> None:
    violations = registry_classification_ambiguity_violations()
    assert not violations, violations


def test_txp5r1_q14_child_runner_new_surface_sentinel_fails_parity() -> None:
    rel = "intergrax/synthetic/child_runner_sentinel_module.py"
    source = """
from intergrax.runtime.execution.child import ChildExecutionRunner

class SyntheticChildHost:
    def __init__(self) -> None:
        self._new_runner = ChildExecutionRunner(ledger=None)
"""
    discovered = discover_child_execution_runner_surfaces_in_source(rel, source)
    sentinel_key = f"{rel}::SyntheticChildHost.__init__"
    assert sentinel_key in discovered
    result = compare_child_runner_surfaces_to_registry(discovered)
    unknown_keys = {f"{path}::{enclosing}" for path, enclosing in result.unknown}
    assert sentinel_key in unknown_keys


def test_txp5r1_q15_child_runner_orphan_registry_row_sentinel() -> None:
    discovered = discover_child_execution_runner_constructor_surfaces()
    if not discovered:
        pytest.fail("no discovered child runner surfaces")
    orphan_key = next(iter(discovered))
    trimmed = frozenset(discovered - {orphan_key})
    result = compare_child_runner_surfaces_to_registry(trimmed)
    path, enclosing = orphan_key.split("::", 1)
    assert (path, enclosing) in result.orphan


def test_txp5r1_q16_child_runner_duplicate_registry_key_sentinel() -> None:
    discovered_keys = frozenset(row.key for row in CHILD_EXECUTION_RUNNER_SURFACE_REGISTRY)
    duplicate_row = CHILD_EXECUTION_RUNNER_SURFACE_REGISTRY[0]
    registry_with_dup = (*CHILD_EXECUTION_RUNNER_SURFACE_REGISTRY, duplicate_row)
    result = compare_discovered_to_registry(discovered_keys, registry_with_dup)
    assert result.duplicate_registry_keys


def test_txp5r1_q17_child_runner_rename_resilience_sentinel() -> None:
    rel = "intergrax/synthetic/renamed_child_runner_host.py"
    source_v1 = """
from intergrax.runtime.execution.child import ChildExecutionRunner

class OriginalHost:
    def __init__(self) -> None:
        self._runner = ChildExecutionRunner()
"""
    source_v2 = """
from intergrax.runtime.execution.child import ChildExecutionRunner

class RenamedHost:
    def boot(self) -> None:
        self._runner = ChildExecutionRunner()
"""
    key_v1 = discover_child_execution_runner_surfaces_in_source(rel, source_v1)
    key_v2 = discover_child_execution_runner_surfaces_in_source(rel, source_v2)
    assert f"{rel}::OriginalHost.__init__" in key_v1
    assert f"{rel}::RenamedHost.boot" in key_v2


def test_txp5r1_q18_wire_host_profile_aware_root_discovery_parity() -> None:
    discovered = discover_wire_host_effective_profile_execution_roots()
    result = compare_profile_aware_wire_host_roots_to_registry(discovered)
    assert not result.unknown, f"unknown profile-aware wire_host roots: {sorted(result.unknown)}"
    assert not result.orphan, f"orphan wire_host registry rows: {sorted(result.orphan)}"
    assert not result.duplicate_registry_keys
    assert len(discovered) == len(PROFILE_AWARE_WIRE_HOST_ROOT_REGISTRY)


def test_txp5r1_q19_profile_aware_roots_forward_child_context_inheritance() -> None:
    for row in PROFILE_AWARE_WIRE_HOST_ROOT_REGISTRY:
        path = Path(__file__).resolve().parents[3] / row.surface_key.split("::", 1)[0]
        source = path.read_text(encoding="utf-8")
        assert profile_aware_root_forwards_child_context_inheritance(source, row.surface_key), row.surface_key


def test_txp5r1_q20_unknown_wire_host_root_without_forwarding_sentinel() -> None:
    rel = "intergrax/synthetic/profile_aware_root_sentinel.py"
    source = """
def build_synthetic_profile_host():
    host_profile = wire_host_effective_profile_execution(env)
    return host_profile
"""
    roots = discover_wire_host_roots_in_source(rel, source)
    assert f"{rel}::build_synthetic_profile_host" in roots
    assert not profile_aware_root_forwards_child_context_inheritance(source, f"{rel}::build_synthetic_profile_host")


def test_txp5r1_q21_global_profile_inheritance_structural_chain() -> None:
    gaps = global_profile_child_inheritance_structural_chain_gaps()
    assert not gaps, gaps


def test_txp5r1_q22_generic_composition_profile_drift_watch() -> None:
    root = Path(__file__).resolve().parents[3]
    violations: list[str] = []
    for watch in GENERIC_COMPOSITION_DRIFT_WATCH_REGISTRY:
        path = root / watch.module_path
        text = path.read_text(encoding="utf-8")
        for marker in watch.forbidden_profile_aware_markers:
            if marker in text:
                violations.append(f"{watch.module_path} contains forbidden marker {marker}")
    assert not violations, violations


def test_txp5r1_q23_production_agent_capability_delegated_subtask_path_documented() -> None:
    path = (
        Path(__file__).resolve().parents[3]
        / "intergrax"
        / "applications"
        / "_shared"
        / "production_agent_capability_runtime.py"
    )
    text = path.read_text(encoding="utf-8")
    assert "build_production_delegated_subtask_child_execution_port" in text
    assert "wire_host_effective_profile_execution" not in text


def test_txp5r1_q24_child_identity_mint_single_production_surface() -> None:
    """Extends TRACE-X-P1 child identity gate (`test_txp1_q04` / child.py mint path)."""
    violations = production_child_identity_mint_surface_violations()
    assert not violations, violations


def test_txp5r1_q25_tenant_isolation_unit_tests_present() -> None:
    path = (
        Path(__file__).resolve().parents[2]
        / "unit"
        / "runtime"
        / "observability"
        / "reconstruction"
        / "test_p5_r1_policy_profile_provenance.py"
    )
    text = path.read_text(encoding="utf-8")
    assert "test_policy_cross_tenant_rejected" in text
    assert "test_profile_cross_tenant_binding_invisible" in text


def test_txp5r1_q26_resume_baseline_evidence_integrity() -> None:
    violations = validate_resume_baseline_evidence(
        expected_q1_start_head=TRACE_X_P5_R1_R1_Q1_START_HEAD,
        expected_q2_start_head=TRACE_X_P5_R1_R1_Q2_START_HEAD,
    )
    assert not violations, violations
    payload = load_resume_baseline_evidence()
    assert payload["schema_version"] == RESUME_EVIDENCE_SCHEMA_VERSION
    assert payload["baseline"]["sha"] == RESUME_BASELINE_SHA
    derived = derive_resume_comparison(payload["baseline"], payload["current"])
    assert derived.conclusion == "PRE_EXISTING_NON_R1_REGRESSION"
    assert payload["comparison"]["conclusion"] == derived.conclusion
    assert set(payload["baseline"]["failed_test_node_ids"]) == set(RESUME_TEST_NODE_IDS)


def test_txp5r1_q27_resume_evidence_wrong_baseline_sha_sentinel() -> None:
    payload = load_resume_baseline_evidence()
    payload = {**payload, "baseline": {**payload["baseline"], "sha": "0" * 40}}
    violations = validate_resume_baseline_payload(
        payload,
        expected_q1_start_head=TRACE_X_P5_R1_R1_Q1_START_HEAD,
    )
    assert any("baseline.sha mismatch" in v for v in violations)


def test_txp5r1_q28_resume_evidence_missing_failed_node_sentinel() -> None:
    payload = load_resume_baseline_evidence()
    payload = {
        **payload,
        "baseline": {**payload["baseline"], "failed_test_node_ids": [RESUME_TEST_NODE_IDS[0]]},
    }
    violations = validate_resume_baseline_payload(
        payload,
        expected_q1_start_head=TRACE_X_P5_R1_R1_Q1_START_HEAD,
    )
    assert any("missing failed test node" in v for v in violations)


def test_txp5r1_q29_resume_evidence_conclusion_signature_mismatch_sentinel() -> None:
    payload = load_resume_baseline_evidence()
    payload = {
        **payload,
        "comparison": {
            **payload["comparison"],
            "same_failure_signatures": True,
            "conclusion": "PRE_EXISTING_NON_R1_REGRESSION",
        },
        "current": {
            **payload["current"],
            "failures": [
                {
                    **payload["current"]["failures"][0],
                    "stable_signature": "tampered|CheckpointResumeValidationError|message",
                },
                payload["current"]["failures"][1],
            ],
        },
    }
    violations = validate_resume_baseline_payload(
        payload,
        expected_q1_start_head=TRACE_X_P5_R1_R1_Q1_START_HEAD,
        expected_q2_start_head=TRACE_X_P5_R1_R1_Q2_START_HEAD,
    )
    assert any("inconsistent with derived" in v for v in violations)


def test_txp5r1_q30_q1_start_head_recorded() -> None:
    assert TRACE_X_P5_R1_R1_Q1_START_HEAD == "a452de39a721cd357be3ba5ecd0c3a6d41b630bd"


def test_txp5r1_q31_q2_start_head_ancestry() -> None:
    assert TRACE_X_P5_R1_R1_Q2_START_HEAD == "538af9ef51a6ca483f607988481ef2794deb87b9"
    subprocess.check_call(
        ["git", "merge-base", "--is-ancestor", TRACE_X_P5_R1_R1_Q2_START_HEAD, "HEAD"],
    )


def test_txp5r1_q32_child_runner_no_production_rebinding() -> None:
    violations = discover_canonical_constructor_rebinding_violations()
    assert not violations, violations


def test_txp5r1_q33_child_runner_alias_import_sentinel_unknown() -> None:
    rel = "intergrax/synthetic/child_runner_alias_import.py"
    source = """
from intergrax.runtime.execution.child import ChildExecutionRunner as CER

class NewHost:
    def __init__(self):
        self.runner = CER()
"""
    discovered = discover_child_execution_runner_surfaces_in_source(rel, source)
    key = f"{rel}::NewHost.__init__"
    assert key in discovered
    result = compare_child_runner_surfaces_to_registry(discovered)
    assert key in {f"{p}::{e}" for p, e in result.unknown}


def test_txp5r1_q34_child_runner_qualified_module_import_sentinel_unknown() -> None:
    rel = "intergrax/synthetic/child_runner_qualified_module.py"
    source = """
import intergrax.runtime.execution.child as child_exec

def build_runner():
    return child_exec.ChildExecutionRunner()
"""
    discovered = discover_child_execution_runner_surfaces_in_source(rel, source)
    key = f"{rel}::build_runner"
    assert key in discovered
    result = compare_child_runner_surfaces_to_registry(discovered)
    assert key in {f"{p}::{e}" for p, e in result.unknown}


def test_txp5r1_q35_child_runner_local_shadow_not_canonical() -> None:
    rel = "intergrax/synthetic/child_runner_local_shadow.py"
    source = """
class ChildExecutionRunner:
    pass

class ShadowHost:
    def __init__(self):
        ChildExecutionRunner()
"""
    discovered = discover_child_execution_runner_surfaces_in_source(rel, source)
    assert not discovered


def test_txp5r1_q36_child_runner_forbidden_rebind_sentinel() -> None:
    rel = "intergrax/synthetic/child_runner_rebind.py"
    source = """
from intergrax.runtime.execution.child import ChildExecutionRunner

class RebindHost:
    def __init__(self):
        Runner = ChildExecutionRunner
        Runner()
"""
    violations = discover_canonical_constructor_rebindings_in_source(rel, source)
    assert violations
    discovered = discover_child_execution_runner_surfaces_in_source(rel, source)
    assert not discovered


def test_txp5r1_q37_child_runner_execution_subpackage_import_sentinel() -> None:
    rel = "intergrax/synthetic/child_runner_subpackage_import.py"
    source = """
from intergrax.runtime.execution import child

def build_runner():
    return child.ChildExecutionRunner()
"""
    discovered = discover_child_execution_runner_surfaces_in_source(rel, source)
    assert f"{rel}::build_runner" in discovered


def test_txp5r1_q38_resume_evidence_baseline_extra_failed_sentinel() -> None:
    payload = load_resume_baseline_evidence()
    extra = "tests/unit/applications/test_effective_profile_revision_adoption.py::test_extra"
    payload = {
        **payload,
        "baseline": {
            **payload["baseline"],
            "failed_test_node_ids": [*payload["baseline"]["failed_test_node_ids"], extra],
            "failed": 3,
        },
    }
    violations = validate_resume_baseline_payload(
        payload,
        expected_q1_start_head=TRACE_X_P5_R1_R1_Q1_START_HEAD,
        expected_q2_start_head=TRACE_X_P5_R1_R1_Q2_START_HEAD,
    )
    assert any("extra failed" in v for v in violations)


def test_txp5r1_q39_resume_evidence_current_missing_failed_sentinel() -> None:
    payload = load_resume_baseline_evidence()
    payload = {
        **payload,
        "current": {
            **payload["current"],
            "failed_test_node_ids": [RESUME_TEST_NODE_IDS[0]],
            "failed": 1,
        },
    }
    violations = validate_resume_baseline_payload(
        payload,
        expected_q1_start_head=TRACE_X_P5_R1_R1_Q1_START_HEAD,
        expected_q2_start_head=TRACE_X_P5_R1_R1_Q2_START_HEAD,
    )
    assert any("missing failed" in v for v in violations)


def test_txp5r1_q40_resume_evidence_failed_count_mismatch_sentinel() -> None:
    payload = load_resume_baseline_evidence()
    payload = {
        **payload,
        "baseline": {**payload["baseline"], "failed": 99},
    }
    violations = validate_resume_baseline_payload(
        payload,
        expected_q1_start_head=TRACE_X_P5_R1_R1_Q1_START_HEAD,
        expected_q2_start_head=TRACE_X_P5_R1_R1_Q2_START_HEAD,
    )
    assert any("failed count mismatch" in v for v in violations)


def test_txp5r1_q41_resume_evidence_exception_type_mismatch_sentinel() -> None:
    payload = load_resume_baseline_evidence()
    failures = list(payload["current"]["failures"])
    failures[0] = {**failures[0], "exception_type": "RuntimeError"}
    payload = {**payload, "current": {**payload["current"], "failures": failures}}
    violations = validate_resume_baseline_payload(
        payload,
        expected_q1_start_head=TRACE_X_P5_R1_R1_Q1_START_HEAD,
        expected_q2_start_head=TRACE_X_P5_R1_R1_Q2_START_HEAD,
    )
    assert any("unexpected exception" in v or "inconsistent with derived" in v for v in violations)


def test_txp5r1_q42_resume_evidence_serialized_same_failed_tests_lie_sentinel() -> None:
    payload = load_resume_baseline_evidence()
    payload = {
        **payload,
        "current": {
            **payload["current"],
            "failed_test_node_ids": [RESUME_TEST_NODE_IDS[0]],
            "failed": 1,
        },
        "comparison": {**payload["comparison"], "same_failed_tests": True},
    }
    violations = validate_resume_baseline_payload(
        payload,
        expected_q1_start_head=TRACE_X_P5_R1_R1_Q1_START_HEAD,
        expected_q2_start_head=TRACE_X_P5_R1_R1_Q2_START_HEAD,
    )
    assert any("same_failed_tests inconsistent" in v for v in violations)


def test_txp5r1_q43_resume_evidence_serialized_regression_lie_sentinel() -> None:
    payload = load_resume_baseline_evidence()
    failures = list(payload["current"]["failures"])
    failures[0] = {**failures[0], "stable_signature": "x|y|z"}
    payload = {
        **payload,
        "current": {**payload["current"], "failures": failures},
        "comparison": {**payload["comparison"], "regression_detected": False},
    }
    violations = validate_resume_baseline_payload(
        payload,
        expected_q1_start_head=TRACE_X_P5_R1_R1_Q1_START_HEAD,
        expected_q2_start_head=TRACE_X_P5_R1_R1_Q2_START_HEAD,
    )
    assert any("regression_detected inconsistent" in v for v in violations)


def test_txp5r1_q44_resume_evidence_serialized_conclusion_lie_sentinel() -> None:
    payload = load_resume_baseline_evidence()
    failures = list(payload["current"]["failures"])
    failures[0] = {**failures[0], "exception_type": "RuntimeError"}
    payload = {
        **payload,
        "current": {**payload["current"], "failures": failures},
        "comparison": {
            **payload["comparison"],
            "conclusion": "PRE_EXISTING_NON_R1_REGRESSION",
        },
    }
    violations = validate_resume_baseline_payload(
        payload,
        expected_q1_start_head=TRACE_X_P5_R1_R1_Q1_START_HEAD,
        expected_q2_start_head=TRACE_X_P5_R1_R1_Q2_START_HEAD,
    )
    assert any("conclusion inconsistent" in v for v in violations)


def test_txp5r1_q45_function_local_direct_import_sentinel_unknown() -> None:
    rel = "intergrax/synthetic/child_runner_fn_local_direct.py"
    source = """
def build_runner():
    from intergrax.runtime.execution.child import ChildExecutionRunner
    return ChildExecutionRunner()
"""
    discovered = discover_child_execution_runner_surfaces_in_source(rel, source)
    key = f"{rel}::build_runner"
    assert key in discovered
    result = compare_child_runner_surfaces_to_registry(discovered)
    assert key in {f"{p}::{e}" for p, e in result.unknown}


def test_txp5r1_q46_function_local_alias_import_sentinel_unknown() -> None:
    rel = "intergrax/synthetic/child_runner_fn_local_alias.py"
    source = """
def build_runner():
    from intergrax.runtime.execution.child import ChildExecutionRunner as CER
    return CER()
"""
    discovered = discover_child_execution_runner_surfaces_in_source(rel, source)
    key = f"{rel}::build_runner"
    assert key in discovered
    result = compare_child_runner_surfaces_to_registry(discovered)
    assert key in {f"{p}::{e}" for p, e in result.unknown}


def test_txp5r1_q47_function_local_qualified_module_import_sentinel_unknown() -> None:
    rel = "intergrax/synthetic/child_runner_fn_local_qualified.py"
    source = """
def build_runner():
    import intergrax.runtime.execution.child as child_exec
    return child_exec.ChildExecutionRunner()
"""
    discovered = discover_child_execution_runner_surfaces_in_source(rel, source)
    key = f"{rel}::build_runner"
    assert key in discovered
    result = compare_child_runner_surfaces_to_registry(discovered)
    assert key in {f"{p}::{e}" for p, e in result.unknown}


def test_txp5r1_q48_function_local_execution_package_child_import_sentinel_unknown() -> None:
    rel = "intergrax/synthetic/child_runner_fn_local_subpackage.py"
    source = """
def build_runner():
    from intergrax.runtime.execution import child
    return child.ChildExecutionRunner()
"""
    discovered = discover_child_execution_runner_surfaces_in_source(rel, source)
    key = f"{rel}::build_runner"
    assert key in discovered
    result = compare_child_runner_surfaces_to_registry(discovered)
    assert key in {f"{p}::{e}" for p, e in result.unknown}


def test_txp5r1_q49_sibling_scope_isolation_sentinel() -> None:
    rel = "intergrax/synthetic/child_runner_sibling_scope.py"
    source = """
def f():
    from intergrax.runtime.execution.child import ChildExecutionRunner as CER
    return CER()

def g():
    return CER()
"""
    discovered = discover_child_execution_runner_surfaces_in_source(rel, source)
    assert f"{rel}::f" in discovered
    assert f"{rel}::g" not in discovered


def test_txp5r1_q50_nested_function_inherits_outer_binding_sentinel_unknown() -> None:
    rel = "intergrax/synthetic/child_runner_nested_inherit.py"
    source = """
def outer():
    from intergrax.runtime.execution.child import ChildExecutionRunner as CER

    def inner():
        return CER()

    CER()
"""
    discovered = discover_child_execution_runner_surfaces_in_source(rel, source)
    assert f"{rel}::inner" in discovered
    assert f"{rel}::outer" in discovered
    source_leak = """
def outer():
    def inner():
        from intergrax.runtime.execution.child import ChildExecutionRunner as CER
        return CER()

    CER()
"""
    leaked = discover_child_execution_runner_surfaces_in_source(rel, source_leak)
    assert f"{rel}::inner" in leaked
    assert f"{rel}::outer" not in leaked


def test_txp5r1_q51_function_local_shadow_reassignment_sentinel() -> None:
    rel = "intergrax/synthetic/child_runner_fn_shadow.py"
    source = """
from intergrax.runtime.execution.child import ChildExecutionRunner

def f():
    ChildExecutionRunner = object()
    return ChildExecutionRunner()
"""
    violations = discover_canonical_constructor_rebindings_in_source(rel, source)
    assert violations
    discovered = discover_child_execution_runner_surfaces_in_source(rel, source)
    assert f"{rel}::f" not in discovered


def test_txp5r1_q52_import_after_call_not_canonical_sentinel() -> None:
    rel = "intergrax/synthetic/child_runner_import_order.py"
    source = """
def f():
    ChildExecutionRunner()
    from intergrax.runtime.execution.child import ChildExecutionRunner
"""
    discovered = discover_child_execution_runner_surfaces_in_source(rel, source)
    assert f"{rel}::f" not in discovered


def test_txp5r1_q53_class_body_canonical_import_forbidden_sentinel() -> None:
    rel = "intergrax/synthetic/child_runner_class_body_import.py"
    source = """
class Host:
    from intergrax.runtime.execution.child import ChildExecutionRunner as CER

    def build(self):
        return CER()
"""
    violations = discover_canonical_constructor_rebindings_in_source(rel, source)
    assert any("class-body" in v for v in violations)
    discovered = discover_child_execution_runner_surfaces_in_source(rel, source)
    assert not discovered


def test_txp5r1_q54_unaliased_dotted_import_semantics_sentinel_unknown() -> None:
    rel = "intergrax/synthetic/child_runner_unaliased_dotted.py"
    source = """
def f():
    import intergrax.runtime.execution.child
    return intergrax.runtime.execution.child.ChildExecutionRunner()

def g():
    import intergrax.runtime.execution.child
    return child.ChildExecutionRunner()
"""
    discovered = discover_child_execution_runner_surfaces_in_source(rel, source)
    assert f"{rel}::f" in discovered
    assert f"{rel}::g" not in discovered
    result = compare_child_runner_surfaces_to_registry(discovered)
    assert f"{rel}::f" in {f"{p}::{e}" for p, e in result.unknown}


def test_txp5r1_q55_q3_start_head_recorded_and_ancestry() -> None:
    assert TRACE_X_P5_R1_R1_Q3_START_HEAD == "c51d04b7849d44d1adee78ad6d4e7eedd8b5ab68"
    subprocess.check_call(
        ["git", "merge-base", "--is-ancestor", TRACE_X_P5_R1_R1_Q3_START_HEAD, "HEAD"],
    )


def _alias_escape_violations(rel: str, source: str) -> list[str]:
    return [
        v
        for v in discover_canonical_constructor_rebindings_in_source(rel, source)
        if "alias-escape" in v or "rebind" in v
    ]


def test_txp5r1_q56_tuple_destructuring_alias_sentinel() -> None:
    rel = "intergrax/synthetic/child_runner_tuple_unpack_alias.py"
    source = """
from intergrax.runtime.execution.child import ChildExecutionRunner

def f():
    Runner, x = ChildExecutionRunner, object()
    Runner()
"""
    assert _alias_escape_violations(rel, source)
    discovered = discover_child_execution_runner_surfaces_in_source(rel, source)
    assert f"{rel}::f" not in discovered


def test_txp5r1_q57_list_destructuring_alias_sentinel() -> None:
    rel = "intergrax/synthetic/child_runner_list_unpack_alias.py"
    source = """
from intergrax.runtime.execution.child import ChildExecutionRunner

def f():
    [Runner, x] = [ChildExecutionRunner, object()]
    Runner()
"""
    assert _alias_escape_violations(rel, source)


def test_txp5r1_q58_container_escape_sentinel() -> None:
    rel = "intergrax/synthetic/child_runner_list_container_escape.py"
    source = """
from intergrax.runtime.execution.child import ChildExecutionRunner

def f():
    factories = [ChildExecutionRunner]
"""
    assert any(
        "alias-escape" in v
        for v in discover_canonical_constructor_rebindings_in_source(rel, source)
    )


def test_txp5r1_q59_dict_escape_sentinel() -> None:
    rel = "intergrax/synthetic/child_runner_dict_escape.py"
    source = """
from intergrax.runtime.execution.child import ChildExecutionRunner

def f():
    factories = {"child": ChildExecutionRunner}
"""
    assert any(
        "alias-escape" in v
        for v in discover_canonical_constructor_rebindings_in_source(rel, source)
    )


def test_txp5r1_q60_conditional_alias_sentinel() -> None:
    rel = "intergrax/synthetic/child_runner_conditional_alias.py"
    source = """
from intergrax.runtime.execution.child import ChildExecutionRunner

class Other:
    pass

def f(flag):
    Runner = ChildExecutionRunner if flag else Other
"""
    assert _alias_escape_violations(rel, source)


def test_txp5r1_q61_qualified_reference_alias_sentinel() -> None:
    rel = "intergrax/synthetic/child_runner_qualified_alias.py"
    source = """
import intergrax.runtime.execution.child as child_exec

def f():
    Runner = child_exec.ChildExecutionRunner
    Runner()
"""
    assert _alias_escape_violations(rel, source)
    discovered = discover_child_execution_runner_surfaces_in_source(rel, source)
    assert f"{rel}::f" not in discovered


def test_txp5r1_q62_direct_call_remains_allowed_sentinel() -> None:
    rel = "intergrax/synthetic/child_runner_direct_call.py"
    source = """
from intergrax.runtime.execution.child import ChildExecutionRunner

def f():
    ChildExecutionRunner()
"""
    discovered = discover_child_execution_runner_surfaces_in_source(rel, source)
    assert f"{rel}::f" in discovered
    assert not discover_canonical_constructor_rebindings_in_source(rel, source)


def test_txp5r1_q63_generic_direct_call_remains_allowed_sentinel() -> None:
    rel = "intergrax/synthetic/child_runner_generic_direct_call.py"
    source = """
from intergrax.runtime.execution.child import ChildExecutionRunner
from typing import TypeVar

A = TypeVar("A")
B = TypeVar("B")

def f():
    ChildExecutionRunner[A, B]()
"""
    discovered = discover_child_execution_runner_surfaces_in_source(rel, source)
    assert f"{rel}::f" in discovered
    assert not discover_canonical_constructor_rebindings_in_source(rel, source)


def test_txp5r1_q64_type_annotation_remains_legal_sentinel() -> None:
    rel = "intergrax/synthetic/child_runner_type_annotation.py"
    source = """
from intergrax.runtime.execution.child import ChildExecutionRunner
from typing import TypeVar

RequestT = TypeVar("RequestT")
ResultT = TypeVar("ResultT")

def accept(runner: ChildExecutionRunner[RequestT, ResultT]) -> None:
    pass
"""
    assert not discover_canonical_constructor_rebindings_in_source(rel, source)


def test_txp5r1_q65_q4_start_head_recorded_and_ancestry() -> None:
    assert TRACE_X_P5_R1_R1_Q4_START_HEAD == "ea1e4947fe66c120bb502885dc912e5f98698339"
    subprocess.check_call(
        ["git", "merge-base", "--is-ancestor", TRACE_X_P5_R1_R1_Q4_START_HEAD, "HEAD"],
    )
