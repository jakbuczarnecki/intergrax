# © Artur Czarnecki. All rights reserved.

"""GR-10-R6-R1 — governed inference composition contract proofs."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from intergrax.contracts.governed_execution_governance_evidence import (
    GovernanceEvidencePersistencePort,
)
from intergrax.runtime.execution.inference_composition import build_governed_inference_executor
from intergrax.runtime.governance.governance_evidence_composition import (
    build_in_memory_governance_evidence_persistence,
)
from intergrax.runtime.policy.policy_engine import PolicyEngine
from testing_support.inference_governance_wiring import (
    build_test_inference_executor_without_evidence,
    governed_inference_executor,
)
from tests.unit.runtime.execution.test_inference_executor import RiskAssessment, StructuredTestAdapter

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_COMPOSITION = _REPO_ROOT / "intergrax" / "runtime" / "execution" / "inference_composition.py"


def test_build_governed_inference_executor_requires_persistence_port_signature() -> None:
    sig = inspect.signature(build_governed_inference_executor)
    param = sig.parameters["governance_evidence_persistence"]
    assert param.default is inspect.Parameter.empty
    assert param.kind is inspect.Parameter.KEYWORD_ONLY


def test_build_governed_inference_executor_no_optional_recorder_parameter() -> None:
    assert "governance_evidence_recorder" not in inspect.signature(
        build_governed_inference_executor
    ).parameters


def test_build_governed_inference_executor_ast_no_default_in_memory_store() -> None:
    source = _COMPOSITION.read_text(encoding="utf-8-sig")
    assert "build_in_memory_governance_evidence_persistence" not in source
    tree = ast.parse(source, filename=str(_COMPOSITION))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "build_governed_inference_executor":
            defaults = node.args.defaults
            assert not defaults
            break
    else:
        pytest.fail("build_governed_inference_executor not found")


def test_governed_inference_executor_wires_custom_port() -> None:
    class _Port(GovernanceEvidencePersistencePort):
        def __init__(self) -> None:
            self.calls = 0

        def persist(self, fact):  # type: ignore[no-untyped-def]
            self.calls += 1
            from intergrax.contracts.governed_execution_governance_evidence import (
                GovernanceEvidencePersistenceOutcome,
            )

            return GovernanceEvidencePersistenceOutcome(
                persisted=True,
                evidence_id=fact.evidence_id,
            )

    port = _Port()
    adapter = StructuredTestAdapter(parsed_output=RiskAssessment(risk="low"))
    executor = build_governed_inference_executor(
        adapter,
        governance_evidence_persistence=port,
        policy_engine=PolicyEngine(),
    )
    assert executor._governance_evidence_recorder is not None
    assert executor._governance_evidence_recorder.persistence is port


def test_test_only_helper_exists_outside_production_builder() -> None:
    adapter = StructuredTestAdapter(parsed_output=RiskAssessment(risk="low"))
    executor = build_test_inference_executor_without_evidence(adapter)
    assert executor._governance_evidence_recorder is None


def test_governed_test_helper_requires_explicit_persistence() -> None:
    sig = inspect.signature(governed_inference_executor)
    assert sig.parameters["governance_evidence_persistence"].default is inspect.Parameter.empty


def test_default_test_persistence_is_explicit_test_choice_not_production_builder() -> None:
    source = ( _REPO_ROOT / "testing_support" / "inference_governance_wiring.py").read_text(
        encoding="utf-8-sig"
    )
    assert "default_test_inference_evidence_persistence" in source
    assert "build_in_memory_governance_evidence_persistence" in source
    composition = _COMPOSITION.read_text(encoding="utf-8-sig")
    assert "build_in_memory_governance_evidence_persistence" not in composition
