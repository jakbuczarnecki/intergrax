# © Artur Czarnecki. All rights reserved.

"""Contract and architecture gates for DS-NPSC-01 coordination semantics."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

import pytest

from intergrax.contracts.decision_artifact_registry import (
    decision_artifact_kind_registry,
    register_decision_artifact_kind,
)
from intergrax.contracts.decision_coordination import (
    DECISION_COORDINATION_ARTIFACT_KIND,
    DecisionCapabilityRequirement,
    DecisionCoordinationContribution,
    DecisionCoordinationSemantic,
    DecisionCoordinationShape,
    DecisionContributionId,
    decision_coordination_artifact,
    decision_coordination_artifact_kind,
    validate_decision_capability_id,
    validate_decision_contribution_id,
)
from intergrax.contracts.decision_identity import (
    DecisionExecutionLineage,
    DecisionIdentity,
    DecisionScope,
    DecisionVersion,
    initial_decision_version,
    mint_decision_id,
)
from intergrax.contracts.decision_record import (
    CandidateDecision,
    DecisionArtifact,
    candidate_decision,
    decision_lineage_ref,
    decision_version_lineage,
)
from intergrax.contracts.execution_identity import (
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
)

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CONTRACT_PATH = _REPO_ROOT / "intergrax" / "contracts" / "decision_coordination.py"
_FORBIDDEN_IMPORT_PREFIXES = (
    "intergrax.agent_distribution",
    "intergrax.runtime.execution",
    "intergrax.runtime.nexus",
)
_FORBIDDEN_AST_NAMES = frozenset(
    {
        "getattr",
        "setattr",
        "hasattr",
        "Any",
    },
)
_FORBIDDEN_FIELD_NAMES = frozenset(
    {
        "agent_id",
        "agent_instance_id",
        "specialist_id",
        "worker_id",
        "provider_instance_id",
        "lease_id",
        "ExecutionId",
        "OrchestrationSlotId",
        "GraphNodeId",
        "GraphExecutor",
        "NexusNode",
        "TaskGroup",
        "Semaphore",
        "scheduler",
        "worker_count",
        "thread_count",
    },
)


@dataclass(frozen=True, slots=True)
class ExamplePayload:
    detail: str


def _capability(capability_id: str) -> DecisionCapabilityRequirement:
    return DecisionCapabilityRequirement(
        capability_id=validate_decision_capability_id(capability_id),
    )


def _contribution(
    contribution_id: str,
    *,
    capability_id: str = "invoice_ocr",
    payload: ExamplePayload | None = None,
) -> DecisionCoordinationContribution[ExamplePayload]:
    return DecisionCoordinationContribution(
        contribution_id=validate_decision_contribution_id(contribution_id),
        capability_requirement=_capability(capability_id),
        payload=payload or ExamplePayload(detail="semantic"),
    )


def _semantic(
    shape: DecisionCoordinationShape,
    contributions: tuple[DecisionCoordinationContribution[ExamplePayload], ...],
) -> DecisionCoordinationSemantic[ExamplePayload]:
    return DecisionCoordinationSemantic(shape=shape, contributions=contributions)


def _collect_imports(path: Path) -> list[tuple[int, str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append((node.lineno, alias.name))
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append((node.lineno, node.module))
    return imports


def _collect_forbidden_ast_usage(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in _FORBIDDEN_AST_NAMES:
            violations.append(f"{path.name}:{node.lineno} references forbidden name {node.id}")
        if isinstance(node, ast.Attribute) and node.attr in _FORBIDDEN_FIELD_NAMES:
            violations.append(f"{path.name}:{node.lineno} references forbidden field {node.attr}")
    return violations


def test_public_import_surface() -> None:
    from intergrax.contracts.decision_coordination import (
        DecisionCoordinationSemantic as ImportedSemantic,
    )

    assert ImportedSemantic is DecisionCoordinationSemantic


def test_valid_single_shape() -> None:
    semantic = _semantic(
        DecisionCoordinationShape.SINGLE,
        (_contribution("contrib-a"),),
    )
    assert semantic.shape is DecisionCoordinationShape.SINGLE
    assert len(semantic.contributions) == 1


def test_valid_fan_out_shape() -> None:
    semantic = _semantic(
        DecisionCoordinationShape.FAN_OUT,
        (
            _contribution("contrib-a", capability_id="invoice_ocr"),
            _contribution("contrib-b", capability_id="fraud_analysis"),
        ),
    )
    assert semantic.shape is DecisionCoordinationShape.FAN_OUT
    assert len(semantic.contributions) == 2


def test_empty_contributions_rejected() -> None:
    with pytest.raises(ValueError, match="contributions must be non-empty"):
        DecisionCoordinationSemantic(
            shape=DecisionCoordinationShape.SINGLE,
            contributions=(),
        )


def test_single_cardinality_rejected() -> None:
    with pytest.raises(ValueError, match="exactly one contribution"):
        _semantic(
            DecisionCoordinationShape.SINGLE,
            (
                _contribution("contrib-a"),
                _contribution("contrib-b"),
            ),
        )


def test_fan_out_cardinality_rejected() -> None:
    with pytest.raises(ValueError, match="at least two contributions"):
        _semantic(
            DecisionCoordinationShape.FAN_OUT,
            (_contribution("contrib-a"),),
        )


def test_duplicate_contribution_id_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate contribution_id"):
        _semantic(
            DecisionCoordinationShape.FAN_OUT,
            (
                _contribution("contrib-a"),
                _contribution("contrib-a", capability_id="catalog_lookup"),
            ),
        )


def test_stable_contribution_identity() -> None:
    first = _contribution("contrib-a")
    second = _contribution("contrib-a", payload=ExamplePayload(detail="semantic"))
    assert first.contribution_id == second.contribution_id
    assert first.contribution_id == DecisionContributionId("contrib-a")


def test_capability_requirement_preserved_through_artifact() -> None:
    contribution = _contribution("contrib-a", capability_id="invoice_ocr")
    semantic = _semantic(DecisionCoordinationShape.SINGLE, (contribution,))
    artifact = decision_coordination_artifact(semantic)
    preserved = artifact.content.contributions[0].capability_requirement
    assert preserved.capability_id == validate_decision_capability_id("invoice_ocr")


def test_payload_preserved_through_decision_artifact() -> None:
    payload = ExamplePayload(detail="typed-value")
    contribution = _contribution("contrib-a", payload=payload)
    semantic = _semantic(DecisionCoordinationShape.SINGLE, (contribution,))
    artifact: DecisionArtifact[DecisionCoordinationSemantic[ExamplePayload]] = (
        decision_coordination_artifact(semantic)
    )
    assert artifact.content.contributions[0].payload == payload
    assert artifact.content.contributions[0].payload.detail == "typed-value"


def test_decision_artifact_integration() -> None:
    semantic = _semantic(DecisionCoordinationShape.SINGLE, (_contribution("contrib-a"),))
    artifact = decision_coordination_artifact(semantic)
    assert artifact.kind == decision_coordination_artifact_kind()
    assert artifact.content.shape is DecisionCoordinationShape.SINGLE


def test_candidate_decision_integration_and_lineage() -> None:
    identity = DecisionIdentity(
        decision_id=mint_decision_id(),
        version=initial_decision_version(),
        scope=DecisionScope(namespace="coordination", subject="case-1"),
        tenant_id="tenant-a",
        execution=DecisionExecutionLineage(
            task_id=mint_task_id(),
            run_id=mint_run_id(),
            attempt_id=mint_attempt_id(),
            execution_id=mint_execution_id(),
        ),
    )
    semantic = _semantic(DecisionCoordinationShape.SINGLE, (_contribution("contrib-a"),))
    candidate: CandidateDecision[DecisionCoordinationSemantic[ExamplePayload]] = (
        candidate_decision(
            identity=identity,
            artifact_kind=decision_coordination_artifact_kind(),
            payload=semantic,
            lineage=decision_version_lineage(
                current=decision_lineage_ref(DecisionVersion(1)),
            ),
        )
    )
    assert candidate.artifact.content.contributions[0].contribution_id == DecisionContributionId(
        "contrib-a",
    )
    assert candidate.lineage.current.version.value == 1


def test_npsc_projection_readiness_without_npsc_imports() -> None:
    semantic = _semantic(
        DecisionCoordinationShape.FAN_OUT,
        (
            _contribution("contrib-a", capability_id="invoice_ocr", payload=ExamplePayload("a")),
            _contribution("contrib-b", capability_id="fraud_analysis", payload=ExamplePayload("b")),
        ),
    )
    artifact = decision_coordination_artifact(semantic)
    projections = [
        (
            artifact.content.shape,
            contribution.contribution_id,
            contribution.capability_requirement.capability_id,
            contribution.payload,
        )
        for contribution in artifact.content.contributions
    ]
    assert projections[0][0] is DecisionCoordinationShape.FAN_OUT
    assert projections[0][1] == DecisionContributionId("contrib-a")
    assert projections[0][2] == validate_decision_capability_id("invoice_ocr")
    assert projections[0][3].detail == "a"


def test_artifact_kind_constant_and_registry() -> None:
    assert DECISION_COORDINATION_ARTIFACT_KIND == "decision.coordination"
    kind = decision_coordination_artifact_kind()
    registry = register_decision_artifact_kind(decision_artifact_kind_registry(), kind)
    assert kind in registry.kinds


def test_contribution_id_validation_fail_closed() -> None:
    with pytest.raises(TypeError):
        validate_decision_contribution_id(1)
    with pytest.raises(ValueError):
        validate_decision_contribution_id("")
    with pytest.raises(ValueError):
        validate_decision_contribution_id("   ")
    with pytest.raises(ValueError):
        validate_decision_contribution_id(" leading")


@pytest.mark.gate
def test_decision_coordination_architecture_gate_no_foreign_imports() -> None:
    violations: list[str] = []
    for lineno, module in _collect_imports(_CONTRACT_PATH):
        if any(
            module == prefix or module.startswith(f"{prefix}.")
            for prefix in _FORBIDDEN_IMPORT_PREFIXES
        ):
            violations.append(f"{_CONTRACT_PATH.name}:{lineno} imports {module}")
    assert not violations, "\n".join(violations)


@pytest.mark.gate
def test_decision_coordination_architecture_gate_no_forbidden_patterns() -> None:
    violations = _collect_forbidden_ast_usage(_CONTRACT_PATH)
    assert not violations, "\n".join(violations)
