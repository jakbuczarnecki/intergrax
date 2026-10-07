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
from tests.qualification.trace_x._trace_x_p5_r1_support import (
    TRACE_X_P5_R1_START_HEAD,
    child_runner_profile_resolution_import_violations,
    discover_profile_aware_environment_host_roots,
    neutral_revision_ref_shadow_grammar_violations,
    profile_aware_orchestration_spec_wiring_gaps,
    profile_child_inheritance_has_production_caller,
    reconstruction_profile_fallback_sentinels,
    reconstruction_runtime_import_violations,
)

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
