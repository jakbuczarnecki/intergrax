# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from intergrax.contracts.capability_catalog import CapabilityKind, CapabilitySourceKind
from intergrax.contracts.capability_catalog.identity_key import CapabilityIdentityKey
from intergrax.contracts.capability_qualification.configured_capability_execution_subject import (
    ConfiguredCapabilityExecutionSubject,
    derive_configured_capability_binding_operation_id,
    derive_configured_capability_execution_operation_id,
    derive_configuration_adoption_identity,
)
from intergrax.contracts.execution_identity import TaskId
from intergrax.contracts.tools.marketplace_tool_execution_intent import (
    MarketplaceToolExecutionProvenanceKind,
    SCHEMA_MARKETPLACE_TOOL_EXECUTION_INTENT_V2,
)
from intergrax.contracts.tools.qualified_marketplace_tool_execution_intent import (
    MarketplaceToolExecutionIntentRepository,
    QualifiedMarketplaceToolExecutionIntentWriteOutcome,
    QualifiedMarketplaceToolExecutionIntentWriteResult,
)
from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.tools.marketplace_configured_tool_execution_intent_preparation import (
    MarketplaceConfiguredToolExecutionIntentPreparation,
    MarketplaceConfiguredToolExecutionIntentPreparationOutcome,
    MarketplaceConfiguredToolExecutionIntentPreparationRequest,
)
from intergrax.tools.marketplace_tool_execution_routing import (
    derive_marketplace_configured_tool_execution_intent_target_correlation,
)
from intergrax.tools.qualified_marketplace_tool_execution_intent_repository import (
    DocumentStoreQualifiedMarketplaceToolExecutionIntentRepository,
)

pytestmark = pytest.mark.unit

_TENANT = "tenant-a"
_TASK_ID = TaskId("task_00000000000000000000000000000001")
_NOW = datetime(2026, 3, 20, 12, 0, 0, tzinfo=UTC)


def _identity() -> CapabilityIdentityKey:
    return CapabilityIdentityKey(
        kind=CapabilityKind.TOOL,
        source_id="official.marketplace",
        source_kind=CapabilitySourceKind.OFFICIAL,
        logical_id="tools.database.relational",
    )


def _subject(**overrides: object) -> ConfiguredCapabilityExecutionSubject:
    fingerprint = "fp-configured-1"
    recovery = "recovery-1"
    decision = "decision-1"
    base = {
        "tenant_id": _TENANT,
        "worker_need_id": "need-1",
        "recovery_decision_id": recovery,
        "decision_id": decision,
        "capability_identity": _identity(),
        "configuration_adoption_identity": derive_configuration_adoption_identity(
            recovery_decision_id=recovery,
            decision_id=decision,
            configuration_fingerprint=fingerprint,
        ),
        "configuration_fingerprint": fingerprint,
        "selected_operations": ("database.query",),
    }
    base.update(overrides)
    return ConfiguredCapabilityExecutionSubject(**base)


def _binding_operation_id(subject: ConfiguredCapabilityExecutionSubject) -> str:
    execution_op = derive_configured_capability_execution_operation_id(
        recovery_decision_id=subject.recovery_decision_id,
        decision_id=subject.decision_id,
    )
    return derive_configured_capability_binding_operation_id(
        configured_execution_operation_id=execution_op,
        subject_reference=subject.subject_reference,
    )


def _request(subject: ConfiguredCapabilityExecutionSubject) -> (
    MarketplaceConfiguredToolExecutionIntentPreparationRequest
):
    binding_id = _binding_operation_id(subject)
    return MarketplaceConfiguredToolExecutionIntentPreparationRequest(
        subject=subject,
        execution_request_id="exec-req-configured-1",
        binding_operation_id=binding_id,
        tenant_id=_TENANT,
        task_id=_TASK_ID,
        worker_need_id="need-1",
    )


class _RecordingRepository(MarketplaceToolExecutionIntentRepository):
    def __init__(self) -> None:
        self.recorded: list[object] = []
        self._inner = DocumentStoreQualifiedMarketplaceToolExecutionIntentRepository(
            InMemoryDocumentStore(),
        )

    def record(self, intent):
        self.recorded.append(intent)
        return self._inner.record(intent)

    def get(self, *, execution_request_id: str):
        return self._inner.get(execution_request_id=execution_request_id)


def test_valid_configured_subject_writes_canonical_v2_intent() -> None:
    subject = _subject()
    repo = _RecordingRepository()
    preparation = MarketplaceConfiguredToolExecutionIntentPreparation(intent_repository=repo)
    result = preparation.prepare(_request(subject))
    assert result.outcome is MarketplaceConfiguredToolExecutionIntentPreparationOutcome.CREATED
    assert len(repo.recorded) == 1
    intent = repo.recorded[0]
    assert intent.schema_version == SCHEMA_MARKETPLACE_TOOL_EXECUTION_INTENT_V2


def test_capability_identity_exactly_once_on_intent() -> None:
    subject = _subject()
    repo = _RecordingRepository()
    preparation = MarketplaceConfiguredToolExecutionIntentPreparation(intent_repository=repo)
    preparation.prepare(_request(subject))
    intent = repo.recorded[0]
    assert intent.capability_identity == subject.capability_identity
    provenance_dump = intent.provenance.model_dump()
    assert "capability_identity" not in provenance_dump


def test_provenance_kind_configured() -> None:
    subject = _subject()
    repo = DocumentStoreQualifiedMarketplaceToolExecutionIntentRepository(InMemoryDocumentStore())
    preparation = MarketplaceConfiguredToolExecutionIntentPreparation(intent_repository=repo)
    preparation.prepare(_request(subject))
    stored = repo.get(execution_request_id="exec-req-configured-1")
    assert stored is not None
    assert stored.provenance.provenance_kind is MarketplaceToolExecutionProvenanceKind.CONFIGURED


def test_tenant_preserved() -> None:
    subject = _subject()
    repo = _RecordingRepository()
    preparation = MarketplaceConfiguredToolExecutionIntentPreparation(intent_repository=repo)
    preparation.prepare(_request(subject))
    assert repo.recorded[0].tenant_id == _TENANT


def test_selected_operation_preserved() -> None:
    subject = _subject(selected_operations=("database.execute",))
    repo = _RecordingRepository()
    preparation = MarketplaceConfiguredToolExecutionIntentPreparation(intent_repository=repo)
    preparation.prepare(_request(subject))
    assert repo.recorded[0].selected_operation == "database.execute"


def test_deterministic_target_binding_correlation() -> None:
    subject = _subject()
    binding_id = _binding_operation_id(subject)
    repo = _RecordingRepository()
    preparation = MarketplaceConfiguredToolExecutionIntentPreparation(intent_repository=repo)
    preparation.prepare(_request(subject))
    intent = repo.recorded[0]
    assert intent.execution_target_correlation == (
        derive_marketplace_configured_tool_execution_intent_target_correlation(binding_id)
    )


def test_repository_receives_marketplace_tool_execution_intent_type() -> None:
    from intergrax.contracts.tools.marketplace_tool_execution_intent import (
        MarketplaceToolExecutionIntent,
    )

    subject = _subject()
    repo = _RecordingRepository()
    preparation = MarketplaceConfiguredToolExecutionIntentPreparation(intent_repository=repo)
    preparation.prepare(_request(subject))
    assert isinstance(repo.recorded[0], MarketplaceToolExecutionIntent)


def test_no_configured_only_repository_module() -> None:
    from pathlib import Path

    tools_root = Path(__file__).resolve().parents[2] / "intergrax" / "tools"
    assert not (tools_root / "configured_marketplace_tool_execution_intent_repository.py").is_file()


def test_invalid_capability_identity_fails_closed() -> None:
    skill_identity = CapabilityIdentityKey(
        kind=CapabilityKind.SKILL,
        source_id="skills.local",
        source_kind=CapabilitySourceKind.LOCAL,
        logical_id="skills.data.query",
    )
    subject = _subject(capability_identity=skill_identity)
    repo = _RecordingRepository()
    preparation = MarketplaceConfiguredToolExecutionIntentPreparation(intent_repository=repo)
    result = preparation.prepare(_request(subject))
    assert result.outcome is (
        MarketplaceConfiguredToolExecutionIntentPreparationOutcome.INTEGRITY_FAILURE
    )
    assert repo.recorded == []


def test_no_provider_object_in_intent_or_provenance() -> None:
    subject = _subject()
    repo = _RecordingRepository()
    preparation = MarketplaceConfiguredToolExecutionIntentPreparation(intent_repository=repo)
    preparation.prepare(_request(subject))
    intent = repo.recorded[0]
    assert intent.provenance.model_dump().keys() == {
        "provenance_kind",
        "recovery_decision_id",
        "acquisition_decision_id",
        "configured_binding_operation_id",
        "configured_execution_operation_id",
        "configuration_adoption_identity",
    }


def test_identical_rerecord_is_stable() -> None:
    subject = _subject()
    store = InMemoryDocumentStore()
    repo = DocumentStoreQualifiedMarketplaceToolExecutionIntentRepository(store)
    preparation = MarketplaceConfiguredToolExecutionIntentPreparation(intent_repository=repo)
    first = preparation.prepare(_request(subject))
    second = preparation.prepare(_request(subject))
    assert first.outcome is MarketplaceConfiguredToolExecutionIntentPreparationOutcome.CREATED
    assert second.outcome is (
        MarketplaceConfiguredToolExecutionIntentPreparationOutcome.ALREADY_RECORDED_IDENTICAL
    )
