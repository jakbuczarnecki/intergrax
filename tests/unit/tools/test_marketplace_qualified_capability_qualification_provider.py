# © Artur Czarnecki. All rights reserved.

"""S24-GAP-02-P2 marketplace tool qualification provider tests."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from intergrax.contracts.capability_acquisition.acquisition_evidence import (
    CapabilityAcquisitionEvidence,
)
from intergrax.contracts.capability_acquisition.acquisition_outcome import (
    CapabilityAcquisitionOutcome,
)
from intergrax.contracts.capability_acquisition.acquisition_reason_code import (
    CapabilityAcquisitionReasonCode,
)
from intergrax.contracts.capability_acquisition.acquisition_result import (
    CapabilityAcquisitionResult,
)
from intergrax.contracts.capability_catalog import (
    CapabilityDiscoveryIdentity,
    CapabilityKind,
    CapabilityLogicalIdentity,
    CapabilityReleaseIdentity,
    CapabilitySourceIdentity,
    CapabilitySourceKind,
)
from intergrax.contracts.capability_qualification.qualification_outcome import (
    CapabilityQualificationOutcome,
)
from intergrax.contracts.capability_qualification.qualification_request import (
    CapabilityQualificationRequest,
    derive_capability_qualification_request_id,
)
from intergrax.contracts.marketplace.handoff_traceability import (
    CapabilityHandoffConsumerTarget,
)
from intergrax.contracts.tools.marketplace_handoff_reference import (
    derive_marketplace_gap_tool_handoff_id,
    marketplace_domain_handoff_reference,
)
from intergrax.contracts.tools.marketplace_qualified_capability import (
    MarketplaceQualifiedToolStage,
)
from intergrax.contracts.tools.marketplace_qualified_tool_stage_context import (
    MarketplaceQualifiedToolStageContext,
)
from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.tools.marketplace_qualified_capability_qualification_provider import (
    MarketplaceToolCapabilityQualificationProvider,
)
from intergrax.tools.marketplace_qualified_capability_staging import (
    DocumentStoreMarketplaceQualifiedToolStageRepository,
)
from intergrax.tools.marketplace_qualified_tool_stage_context_association import (
    DocumentStoreMarketplaceQualifiedToolStageContextAssociationRepository,
)
from intergrax.tools.marketplace_qualified_tool_stage_context_resolver import (
    MarketplaceQualifiedToolStageContextResolverImpl,
)

pytestmark = pytest.mark.unit

_NOW = datetime(2026, 3, 26, 12, 0, tzinfo=UTC)
_STRATEGY = "marketplace.gap_acquisition.v1"
_SOURCE = CapabilitySourceIdentity(
    source_id="official.gap02",
    source_kind=CapabilitySourceKind.OFFICIAL,
)


def _release() -> CapabilityReleaseIdentity:
    return CapabilityReleaseIdentity(
        discovery=CapabilityDiscoveryIdentity(
            kind=CapabilityKind.TOOL,
            source=_SOURCE,
            logical=CapabilityLogicalIdentity(
                kind=CapabilityKind.TOOL,
                logical_id="tools.qual",
            ),
        ),
        publisher="publisher",
        version_label="1.0.0",
        content_digest="sha256:qual",
        package_reference="pkg://qual",
    )


def _stack(
    *,
    tenant_id: str = "tenant-1",
    acquisition_id: str = "acq-1",
) -> tuple[
    MarketplaceToolCapabilityQualificationProvider,
    str,
    str,
]:
    store = InMemoryDocumentStore()
    stage_repo = DocumentStoreMarketplaceQualifiedToolStageRepository(store)
    assoc_repo = DocumentStoreMarketplaceQualifiedToolStageContextAssociationRepository(
        store,
    )
    resolver = MarketplaceQualifiedToolStageContextResolverImpl(assoc_repo)
    provider = MarketplaceToolCapabilityQualificationProvider(
        stage_repository=stage_repo,
        context_resolver=resolver,
    )
    handoff_id = derive_marketplace_gap_tool_handoff_id(
        tenant_id=tenant_id,
        operation_id=acquisition_id,
    )
    assoc_repo.record(
        MarketplaceQualifiedToolStageContext(
            handoff_id=handoff_id,
            tenant_id=tenant_id,
            acquisition_request_id=acquisition_id,
        ),
    )
    stage_repo.stage(
        MarketplaceQualifiedToolStage(
            handoff_id=handoff_id,
            tenant_id=tenant_id,
            selected_release=_release(),
            discovery_correlation_id="discovery-1",
            selection_id=f"marketplace-gap-selection:{acquisition_id}",
            consumer_target=CapabilityHandoffConsumerTarget.TOOL_DOMAIN,
            downstream_consumer_id="tool.qualification_staging.v1",
            recorded_at=_NOW,
        ),
    )
    domain_ref = marketplace_domain_handoff_reference(handoff_id)
    return provider, acquisition_id, domain_ref


def _qualification_request(
    *,
    acquisition_id: str,
    domain_ref: str,
    strategy_id: str = _STRATEGY,
    artifact_reference: str | None = None,
    evidence_ref: str | None = None,
) -> CapabilityQualificationRequest:
    acquisition = CapabilityAcquisitionResult(
        request_id=acquisition_id,
        gap_id="gap-1",
        strategy_id=strategy_id,
        outcome=CapabilityAcquisitionOutcome.SUCCEEDED,
        reason_code=CapabilityAcquisitionReasonCode.NONE,
        started_at=_NOW,
        completed_at=_NOW,
        evidence=CapabilityAcquisitionEvidence(
            domain_handoff_reference=domain_ref,
            artifact_reference=artifact_reference,
            evidence_ref=evidence_ref,
        ),
    )
    return CapabilityQualificationRequest(
        qualification_request_id=derive_capability_qualification_request_id(
            acquisition_request_id=acquisition_id,
            qualification_nonce="nonce-1",
        ),
        qualification_nonce="nonce-1",
        acquisition_request_id=acquisition_id,
        gap_id="gap-1",
        strategy_id=strategy_id,
        acquisition_result=acquisition,
        requested_at=_NOW,
    )


def test_marketplace_tool_subject_qualified() -> None:
    provider, acquisition_id, domain_ref = _stack()
    result = provider.qualify(_qualification_request(acquisition_id=acquisition_id, domain_ref=domain_ref))
    assert result.outcome is CapabilityQualificationOutcome.QUALIFIED
    assert result.evidence is not None
    assert result.evidence.domain_handoff_reference == domain_ref


def test_stage_missing_unavailable() -> None:
    store = InMemoryDocumentStore()
    assoc_repo = DocumentStoreMarketplaceQualifiedToolStageContextAssociationRepository(
        store,
    )
    resolver = MarketplaceQualifiedToolStageContextResolverImpl(assoc_repo)
    provider = MarketplaceToolCapabilityQualificationProvider(
        stage_repository=DocumentStoreMarketplaceQualifiedToolStageRepository(store),
        context_resolver=resolver,
    )
    acquisition_id = "acq-missing-stage"
    handoff_id = derive_marketplace_gap_tool_handoff_id(
        tenant_id="tenant-1",
        operation_id=acquisition_id,
    )
    assoc_repo.record(
        MarketplaceQualifiedToolStageContext(
            handoff_id=handoff_id,
            tenant_id="tenant-1",
            acquisition_request_id=acquisition_id,
        ),
    )
    domain_ref = marketplace_domain_handoff_reference(handoff_id)
    result = provider.qualify(
        _qualification_request(acquisition_id=acquisition_id, domain_ref=domain_ref),
    )
    assert result.outcome is CapabilityQualificationOutcome.UNAVAILABLE


def test_wrong_strategy_not_supported() -> None:
    provider, acquisition_id, domain_ref = _stack()
    request = _qualification_request(
        acquisition_id=acquisition_id,
        domain_ref=domain_ref,
        strategy_id="other.strategy",
    )
    assert provider.supports(request) is False


def test_canonical_evidence_ref_supported_and_qualified() -> None:
    provider, acquisition_id, domain_ref = _stack()
    request = _qualification_request(
        acquisition_id=acquisition_id,
        domain_ref=domain_ref,
        evidence_ref="marketplace-gap-listing:corr",
    )
    assert provider.supports(request) is True
    result = provider.qualify(request)
    assert result.outcome is CapabilityQualificationOutcome.QUALIFIED


def test_artifact_only_not_supported() -> None:
    provider, acquisition_id, domain_ref = _stack()
    request = _qualification_request(
        acquisition_id=acquisition_id,
        domain_ref=domain_ref,
        artifact_reference="artifact://x",
    )
    assert provider.supports(request) is False


def test_no_activation_dependencies() -> None:
    path = Path(
        "intergrax/tools/marketplace_qualified_capability_qualification_provider.py",
    )
    source = path.read_text(encoding="utf-8")
    for token in ("ToolRegistry", "ToolRuntime", "DynamicToolAcquisition", "ExecutionEngine"):
        assert token not in source
