# © Artur Czarnecki. All rights reserved.

"""ERL-DIAG-001A public reliability diagnostics contract tests."""

from __future__ import annotations

import ast
import inspect
import typing
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from intergrax.contracts.enterprise_reliability.diagnostics import (
    AutomationSafetyHint,
    ExternalEffectReliabilityDiagnosticEmitter,
    ExternalEffectReliabilityObservation,
    ExternalEffectReliabilityObservationValidationError,
    ExternalEffectReliabilitySignalKind,
    MAX_RELIABILITY_DIAGNOSTIC_TRACE_REFS,
    NullExternalEffectReliabilityDiagnosticEmitter,
    ReliabilityDiagnosticArtifactRefs,
    ReliabilityDiagnosticCorrelation,
    SCHEMA_EXTERNAL_EFFECT_RELIABILITY_OBSERVATION_V1,
    __all__ as diagnostics_public_exports,
)
from intergrax.contracts.execution_identity import (
    validate_attempt_id,
    validate_execution_id,
    validate_run_id,
    validate_task_id,
)

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DIAGNOSTICS_PKG = _REPO_ROOT / "intergrax" / "contracts" / "enterprise_reliability" / "diagnostics"
_FORBIDDEN_IMPORT_PREFIXES = (
    "intergrax.runtime",
    "platform_proofs",
    "applications",
)
_FORBIDDEN_AST_NAMES = frozenset({"getattr", "setattr", "hasattr", "vars", "Any"})


def _default_correlation() -> ReliabilityDiagnosticCorrelation:
    return ReliabilityDiagnosticCorrelation(
        tenant_id="tenant-a",
        correlation_id="corr-1",
        reliability_case_id="case-1",
        external_effect_contract_id="contract-1",
    )


def _observation(
    *,
    observation_id: str = "obs-1",
    tenant_id: str = "tenant-a",
    signal_kind: ExternalEffectReliabilitySignalKind = (
        ExternalEffectReliabilitySignalKind.UNCERTAINTY_ADMITTED
    ),
    reliability_case_id: str = "case-1",
    correlation: ReliabilityDiagnosticCorrelation | None = None,
    artifact_refs: ReliabilityDiagnosticArtifactRefs | None = None,
    trace_refs: tuple[str, ...] = (),
) -> ExternalEffectReliabilityObservation:
    from intergrax.contracts.enterprise_reliability.case_lifecycle import (
        ReliabilityCaseLifecycleState,
    )

    return ExternalEffectReliabilityObservation(
        observation_id=observation_id,
        tenant_id=tenant_id,
        signal_kind=signal_kind,
        recorded_at=datetime(2026, 1, 1, tzinfo=UTC),
        reliability_case_id=reliability_case_id,
        correlation=correlation or _default_correlation(),
        lifecycle_state=ReliabilityCaseLifecycleState.UNKNOWN_DETECTED,
        artifact_refs=artifact_refs or ReliabilityDiagnosticArtifactRefs(),
        execution_safety_hint=AutomationSafetyHint.UNKNOWN,
        trace_refs=trace_refs,
    )


def test_observation_valid_construction() -> None:
    obs = _observation()
    assert obs.schema_version == SCHEMA_EXTERNAL_EFFECT_RELIABILITY_OBSERVATION_V1
    assert obs.observation_id == "obs-1"


def test_empty_observation_id_rejected() -> None:
    with pytest.raises(ValidationError):
        _observation(observation_id="")


def test_whitespace_observation_id_rejected() -> None:
    with pytest.raises(ValidationError):
        _observation(observation_id="   ")


def test_tenant_mismatch_rejected() -> None:
    with pytest.raises(ValidationError, match="tenant_id"):
        _observation(tenant_id="other-tenant")


def test_reliability_case_mismatch_rejected() -> None:
    with pytest.raises(ValidationError, match="reliability_case_id"):
        _observation(reliability_case_id="case-other")


def test_observation_validation_error_is_value_error_subclass() -> None:
    assert issubclass(ExternalEffectReliabilityObservationValidationError, ValueError)


def test_signal_kind_stable_values() -> None:
    assert ExternalEffectReliabilitySignalKind.UNCERTAINTY_ADMITTED.value == "UNCERTAINTY_ADMITTED"
    assert len(ExternalEffectReliabilitySignalKind) == 10


def test_automation_safety_hint_stable_values() -> None:
    assert AutomationSafetyHint.SAFE.value == "SAFE"
    assert AutomationSafetyHint.UNSAFE.value == "UNSAFE"
    assert AutomationSafetyHint.UNKNOWN.value == "UNKNOWN"


def test_evidence_ref_required_for_reconciliation_signal() -> None:
    with pytest.raises(ValidationError, match="evidence_ref"):
        _observation(
            signal_kind=ExternalEffectReliabilitySignalKind.RECONCILIATION_ATTEMPTED,
            artifact_refs=ReliabilityDiagnosticArtifactRefs(),
        )


def test_external_emitter_implementable() -> None:
    captured: list[ExternalEffectReliabilityObservation] = []

    class _RecordingEmitter:
        def emit(self, observation: ExternalEffectReliabilityObservation) -> None:
            captured.append(observation)

    emitter: ExternalEffectReliabilityDiagnosticEmitter = _RecordingEmitter()
    obs = _observation()
    emitter.emit(obs)
    assert captured == [obs]


def test_null_emitter_satisfies_contract() -> None:
    null = NullExternalEffectReliabilityDiagnosticEmitter()
    null.emit(_observation())


def test_observation_immutable() -> None:
    from pydantic import ValidationError

    obs = _observation()
    with pytest.raises(ValidationError):
        obs.observation_id = "mutated"  # type: ignore[misc]


def test_artifact_refs_immutable() -> None:
    from pydantic import ValidationError

    refs = ReliabilityDiagnosticArtifactRefs(evidence_ref="erl:evidence:1")
    with pytest.raises(ValidationError):
        refs.evidence_ref = "mutated"  # type: ignore[misc]


def test_correlation_identities_remain_separate() -> None:
    correlation = ReliabilityDiagnosticCorrelation(
        tenant_id="tenant-a",
        correlation_id="corr-1",
        reliability_case_id="case-1",
        external_effect_contract_id="contract-1",
        execution_id=validate_execution_id("exec_01234567890123456789012345678901"),
        run_id=validate_run_id("run_01234567890123456789012345678901"),
        task_id=validate_task_id("task_01234567890123456789012345678901"),
        attempt_id=validate_attempt_id("attempt_01234567890123456789012345678901"),
        trace_id="trace-abc",
        idempotency_key="biz-key-1",
    )
    assert correlation.correlation_id == "corr-1"
    assert correlation.reliability_case_id == "case-1"
    assert correlation.external_effect_contract_id == "contract-1"
    assert str(correlation.execution_id).startswith("exec_")


def test_trace_refs_bounded() -> None:
    too_many = tuple(f"trace-{index}" for index in range(MAX_RELIABILITY_DIAGNOSTIC_TRACE_REFS + 1))
    with pytest.raises(ValueError, match="trace_refs exceeds"):
        _observation(trace_refs=too_many)


def test_observation_model_dump_json_roundtrip() -> None:
    obs = _observation(
        signal_kind=ExternalEffectReliabilitySignalKind.EVIDENCE_SUFFICIENT,
        artifact_refs=ReliabilityDiagnosticArtifactRefs(evidence_ref="erl:evidence:1"),
    )
    restored = ExternalEffectReliabilityObservation.model_validate_json(obs.model_dump_json())
    assert restored == obs


def test_public_exports_intentional() -> None:
    expected = {
        "AutomationSafetyHint",
        "ExternalEffectReliabilityDiagnosticEmitter",
        "ExternalEffectReliabilityObservation",
        "ExternalEffectReliabilityObservationValidationError",
        "ExternalEffectReliabilityProblemGroupingStrategy",
        "ExternalEffectReliabilitySignalKind",
        "MAX_RELIABILITY_DIAGNOSTIC_TRACE_REFS",
        "NullExternalEffectReliabilityDiagnosticEmitter",
        "RELIABILITY_CASE_DEFAULT_GROUPING_STRATEGY_ID",
        "RELIABILITY_CASE_DEFAULT_GROUPING_STRATEGY_VERSION",
        "ReliabilityCaseSubjectRef",
        "ReliabilityDiagnosticArtifactRefs",
        "ReliabilityDiagnosticCorrelation",
        "ReliabilityProblemGroupingStrategyId",
        "ReliabilityProblemGroupingStrategyVersion",
        "SCHEMA_AUTOMATION_SAFETY_HINT_V1",
        "SCHEMA_EXTERNAL_EFFECT_RELIABILITY_OBSERVATION_V1",
        "SCHEMA_EXTERNAL_EFFECT_RELIABILITY_SIGNAL_KIND_V1",
        "parse_reliability_diagnostic_occurrence_instance_id",
        "reliability_case_subject_index_token",
        "reliability_correlation_subject_index_token",
        "reliability_diagnostic_occurrence_instance_id",
    }
    assert set(diagnostics_public_exports) == expected


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
            violations.append(f"{path.relative_to(_REPO_ROOT)}:{node.lineno} uses {node.id}")
    return violations


@pytest.mark.gate
def test_diagnostics_package_no_runtime_or_scenario_imports() -> None:
    violations: list[str] = []
    for path in sorted(_DIAGNOSTICS_PKG.glob("*.py")):
        for lineno, module in _collect_imports(path):
            for prefix in _FORBIDDEN_IMPORT_PREFIXES:
                if module == prefix or module.startswith(f"{prefix}."):
                    violations.append(f"{path.name}:{lineno} imports {module}")
    assert not violations, "\n".join(violations)


@pytest.mark.gate
def test_diagnostics_package_no_forbidden_dynamic_patterns() -> None:
    violations: list[str] = []
    for path in sorted(_DIAGNOSTICS_PKG.glob("*.py")):
        violations.extend(_collect_forbidden_ast_usage(path))
    assert not violations, "\n".join(violations)


def test_public_models_do_not_use_any() -> None:
    from pydantic import BaseModel

    from intergrax.contracts.enterprise_reliability import diagnostics as pkg

    for name in diagnostics_public_exports:
        obj = getattr(pkg, name)
        if not (inspect.isclass(obj) and issubclass(obj, BaseModel)):
            continue
        model_cls: type[BaseModel] = obj
        for field_name, field_info in model_cls.model_fields.items():
            annotation = field_info.annotation
            assert typing.Any not in typing.get_args(annotation) and annotation is not typing.Any, (
                f"{name}.{field_name}"
            )


@pytest.mark.gate
def test_single_canonical_observation_definition() -> None:
    contracts_root = _REPO_ROOT / "intergrax" / "contracts"
    hits: list[str] = []
    for path in contracts_root.rglob("*.py"):
        if "enterprise_reliability" in path.parts and "diagnostics" in path.parts:
            continue
        text = path.read_text(encoding="utf-8")
        if "class ExternalEffectReliabilityObservation" in text:
            hits.append(str(path.relative_to(_REPO_ROOT)))
    assert hits == [], f"duplicate observation definitions: {hits}"


@pytest.mark.gate
def test_single_canonical_emitter_definition() -> None:
    contracts_root = _REPO_ROOT / "intergrax" / "contracts"
    protocol_hits: list[str] = []
    for path in contracts_root.rglob("*.py"):
        if "enterprise_reliability" in path.parts and "diagnostics" in path.parts:
            continue
        text = path.read_text(encoding="utf-8")
        if "class ExternalEffectReliabilityDiagnosticEmitter" in text:
            protocol_hits.append(str(path.relative_to(_REPO_ROOT)))
    assert protocol_hits == [], f"duplicate emitter definitions: {protocol_hits}"
