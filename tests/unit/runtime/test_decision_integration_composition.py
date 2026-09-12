# © Artur Czarnecki. All rights reserved.

"""DS-E2E-15J platform composition integration tests."""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import pytest

from intergrax.contracts.decision.integration import (
    ConfiguredDecisionIntegrationCompositionProvider,
    DecisionIntegrationAuditRecord,
    DecisionIntegrationCompositionSpec,
    DecisionIntegrationStatus,
    DecisionSystemIntegrationEngine,
    DecisionSystemIntegrationFactory,
    REFERENCE_DECISION_LIFECYCLE_SOURCE_TYPE,
    ReferenceDecisionLifecycleReference,
    ReferenceEnterpriseLifecycleState,
    SingleLifecycleAdapterProvider,
)
from intergrax.contracts.decision.integration.result import (
    DecisionAdapterMetadata,
    DecisionIntegrationResult,
)
from intergrax.contracts.decision_lifecycle import DecisionLifecycleStage
from intergrax.runtime.decision_integration_composition import (
    compose_decision_system_integration_engine_with_providers,
    default_decision_system_integration,
)
from intergrax.runtime.decision_plugin_composition import (
    compose_decision_system_integration_from_platform,
)
from testing_support.decision_e2e.model_matrix.enterprise_decision_lifecycle.contracts import (
    DecisionLifecycleRecord,
    DecisionLifecycleState,
    DecisionSourceKind,
    DecisionSourceReference,
    DecisionType,
)

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[3]
_COMPOSITION_CONTRACTS_ROOT = (
    _REPO_ROOT / "intergrax" / "contracts" / "decision" / "integration"
)


def _reference_source() -> ReferenceDecisionLifecycleReference:
    record = DecisionLifecycleRecord(
        decision_id="composition-decision-001",
        decision_type=DecisionType.PRODUCTION_MODEL_ROUTING,
        lifecycle_state=DecisionLifecycleState.CREATED,
        created_at=datetime(2026, 1, 15, 12, 0, tzinfo=UTC),
        source_references=(
            DecisionSourceReference(
                source_kind=DecisionSourceKind.MODEL_SELECTION,
                reference_id="sel-1",
            ),
        ),
    )
    return ReferenceDecisionLifecycleReference(
        source_type=REFERENCE_DECISION_LIFECYCLE_SOURCE_TYPE,
        decision_id=record.decision_id,
        lifecycle_state=ReferenceEnterpriseLifecycleState(record.lifecycle_state.value),
        decision_type=record.decision_type.value,
        created_at_iso=record.created_at.isoformat(),
        mapping_version="1",
    )


def test_default_composition_creates_working_engine() -> None:
    engine = default_decision_system_integration()
    result = engine.integrate_lifecycle(_reference_source())

    assert type(engine) is DecisionSystemIntegrationEngine
    assert result.status is DecisionIntegrationStatus.SUCCESS
    assert result.target is not None
    assert result.target.stage is DecisionLifecycleStage.PROPOSAL


@dataclass(frozen=True, slots=True)
class _CustomCompositionLifecycleAdapter:
    @property
    def adapter_id(self) -> str:
        return "composition.custom.lifecycle"

    @property
    def adapter_version(self) -> str:
        return "2.0.0"

    @property
    def mapping_version(self) -> str:
        return "custom-comp-1"

    @property
    def source_type(self) -> str:
        return REFERENCE_DECISION_LIFECYCLE_SOURCE_TYPE

    def integrate_lifecycle(
        self,
        source: ReferenceDecisionLifecycleReference,
    ) -> DecisionIntegrationResult:
        integrated_at = datetime.now(tz=UTC)
        metadata = DecisionAdapterMetadata(
            source_type=source.source_type,
            adapter_id=self.adapter_id,
            adapter_version=self.adapter_version,
            mapping_version=self.mapping_version,
            integrated_at=integrated_at,
        )
        from intergrax.contracts.decision.integration.references import (
            PlatformDecisionLifecycleReference,
        )

        target = PlatformDecisionLifecycleReference(
            reference_decision_id=source.decision_id,
            stage=DecisionLifecycleStage.VERIFICATION,
            transition_index=0,
            mapping_version=self.mapping_version,
        )
        return DecisionIntegrationResult(
            status=DecisionIntegrationStatus.SUCCESS,
            source=source,
            target=target,
            adapter_metadata=metadata,
            detail="custom composition adapter",
        )


def test_custom_adapter_provider_through_factory() -> None:
    custom_provider = SingleLifecycleAdapterProvider(
        lifecycle_adapter=_CustomCompositionLifecycleAdapter(),
    )
    composition = ConfiguredDecisionIntegrationCompositionProvider(
        composition_spec=DecisionIntegrationCompositionSpec(
            active_lifecycle_source_types=frozenset(
                {REFERENCE_DECISION_LIFECYCLE_SOURCE_TYPE},
            ),
            audit_enabled=False,
        ),
        adapter_providers=(custom_provider,),
        audit_provider=None,
    )
    engine = DecisionSystemIntegrationFactory.create_engine(composition)
    result = engine.integrate_lifecycle(_reference_source())

    assert result.status is DecisionIntegrationStatus.SUCCESS
    assert result.adapter_metadata.adapter_id == "composition.custom.lifecycle"


def test_composition_without_adapter_fails_controlled() -> None:
    composition = ConfiguredDecisionIntegrationCompositionProvider(
        composition_spec=DecisionIntegrationCompositionSpec(
            active_lifecycle_source_types=frozenset(),
            audit_enabled=False,
        ),
        adapter_providers=(),
        audit_provider=None,
    )
    engine = DecisionSystemIntegrationFactory.create_engine(composition)
    result = engine.integrate_lifecycle(_reference_source())

    assert result.status is DecisionIntegrationStatus.FAILED


@dataclass
class _RecordingAuditProvider:
    records: list[DecisionIntegrationAuditRecord] = field(default_factory=list)

    def record_integration(self, record: DecisionIntegrationAuditRecord) -> None:
        self.records.append(record)


def test_audit_propagation_through_composition_root() -> None:
    audit = _RecordingAuditProvider()
    engine = compose_decision_system_integration_engine_with_providers(
        adapter_providers=(
            SingleLifecycleAdapterProvider(
                lifecycle_adapter=_CustomCompositionLifecycleAdapter(),
            ),
        ),
        audit_provider=audit,
    )
    result = engine.integrate_lifecycle(_reference_source())

    assert len(audit.records) == 1
    assert audit.records[0].target is result.target
    assert (
        audit.records[0].adapter_metadata.adapter_id == "composition.custom.lifecycle"
    )


def _collect_imports(path: Path, prefix: str) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    hits: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith(prefix):
                    hits.append(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.module.startswith(prefix):
                hits.append(node.module)
    return hits


def test_composition_contracts_do_not_import_runtime_execution() -> None:
    violations: list[str] = []
    for path in sorted(_COMPOSITION_CONTRACTS_ROOT.rglob("*.py")):
        rel = path.relative_to(_REPO_ROOT).as_posix()
        for module in _collect_imports(path, "intergrax.runtime"):
            violations.append(f"{rel}: {module}")
    assert violations == [], (
        "integration contracts must not import runtime:\n"
        + "\n".join(
            violations,
        )
    )

    factory_path = _COMPOSITION_CONTRACTS_ROOT / "factory.py"
    execution_hits = _collect_imports(factory_path, "intergrax.runtime.execution")
    assert execution_hits == []


def test_platform_plugin_composition_entry_matches_default() -> None:
    from_platform = compose_decision_system_integration_from_platform()
    direct = default_decision_system_integration()
    source = _reference_source()

    assert (
        from_platform.integrate_lifecycle(source).status
        == direct.integrate_lifecycle(
            source,
        ).status
    )
