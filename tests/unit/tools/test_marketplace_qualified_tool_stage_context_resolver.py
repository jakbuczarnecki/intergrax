# © Artur Czarnecki. All rights reserved.

"""S24-GAP-02-P2 context resolver tests."""

from __future__ import annotations

import pytest

from intergrax.contracts.tools.marketplace_handoff_reference import (
    derive_marketplace_gap_tool_handoff_id,
    marketplace_domain_handoff_reference,
)
from intergrax.contracts.tools.marketplace_qualified_tool_stage_context import (
    MarketplaceQualifiedToolStageContext,
    MarketplaceQualifiedToolStageContextAssociationUnavailableError,
    MarketplaceQualifiedToolStageContextNotFoundError,
    MarketplaceQualifiedToolStageContextResolverConflictError,
    MarketplaceQualifiedToolStageContextResolverIntegrityError,
    MarketplaceQualifiedToolStageContextResolverNotSupportedError,
    MarketplaceQualifiedToolStageContextResolverUnavailableError,
)
from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.tools.marketplace_qualified_tool_stage_context_association import (
    DocumentStoreMarketplaceQualifiedToolStageContextAssociationRepository,
)
from intergrax.tools.marketplace_qualified_tool_stage_context_resolver import (
    MarketplaceQualifiedToolStageContextResolverImpl,
)

pytestmark = pytest.mark.unit

_STRATEGY = "marketplace.gap_acquisition.v1"


def _resolver() -> tuple[
    MarketplaceQualifiedToolStageContextResolverImpl,
    DocumentStoreMarketplaceQualifiedToolStageContextAssociationRepository,
]:
    store = InMemoryDocumentStore()
    repo = DocumentStoreMarketplaceQualifiedToolStageContextAssociationRepository(store)
    return MarketplaceQualifiedToolStageContextResolverImpl(repo), repo


def test_valid_handoff_resolves_context() -> None:
    resolver, repo = _resolver()
    tenant_id = "tenant-1"
    acquisition_id = "acq-1"
    handoff_id = derive_marketplace_gap_tool_handoff_id(
        tenant_id=tenant_id,
        operation_id=acquisition_id,
    )
    repo.record(
        MarketplaceQualifiedToolStageContext(
            handoff_id=handoff_id,
            tenant_id=tenant_id,
            acquisition_request_id=acquisition_id,
        ),
    )
    ctx = resolver.resolve_for_qualification(
        acquisition_request_id=acquisition_id,
        domain_handoff_reference=marketplace_domain_handoff_reference(handoff_id),
        strategy_id=_STRATEGY,
    )
    assert ctx.tenant_id == tenant_id
    assert ctx.handoff_id == handoff_id


def test_missing_association_not_found() -> None:
    resolver, _repo = _resolver()
    handoff_id = derive_marketplace_gap_tool_handoff_id(
        tenant_id="t",
        operation_id="missing",
    )
    with pytest.raises(MarketplaceQualifiedToolStageContextNotFoundError):
        resolver.resolve_for_qualification(
            acquisition_request_id="missing",
            domain_handoff_reference=marketplace_domain_handoff_reference(handoff_id),
            strategy_id=_STRATEGY,
        )


def test_malformed_reference_integrity() -> None:
    resolver, _repo = _resolver()
    with pytest.raises(MarketplaceQualifiedToolStageContextResolverIntegrityError):
        resolver.resolve_for_qualification(
            acquisition_request_id="acq",
            domain_handoff_reference="not-a-handoff",
            strategy_id=_STRATEGY,
        )


def test_acquisition_id_mismatch_conflict() -> None:
    resolver, repo = _resolver()
    handoff_id = derive_marketplace_gap_tool_handoff_id(
        tenant_id="tenant-1",
        operation_id="acq-1",
    )
    repo.record(
        MarketplaceQualifiedToolStageContext(
            handoff_id=handoff_id,
            tenant_id="tenant-1",
            acquisition_request_id="acq-1",
        ),
    )
    with pytest.raises(MarketplaceQualifiedToolStageContextResolverConflictError):
        resolver.resolve_for_qualification(
            acquisition_request_id="acq-2",
            domain_handoff_reference=marketplace_domain_handoff_reference(handoff_id),
            strategy_id=_STRATEGY,
        )


def test_tenant_modified_in_store_integrity() -> None:
    resolver, repo = _resolver()
    acquisition_id = "shared-acq"
    handoff_a = derive_marketplace_gap_tool_handoff_id(
        tenant_id="tenant-a",
        operation_id=acquisition_id,
    )
    repo.record(
        MarketplaceQualifiedToolStageContext(
            handoff_id=handoff_a,
            tenant_id="tenant-evil",
            acquisition_request_id=acquisition_id,
        ),
    )
    with pytest.raises(MarketplaceQualifiedToolStageContextResolverIntegrityError):
        resolver.resolve_for_qualification(
            acquisition_request_id=acquisition_id,
            domain_handoff_reference=marketplace_domain_handoff_reference(handoff_a),
            strategy_id=_STRATEGY,
        )


def test_strategy_not_supported() -> None:
    resolver, _repo = _resolver()
    with pytest.raises(MarketplaceQualifiedToolStageContextResolverNotSupportedError):
        resolver.resolve_for_qualification(
            acquisition_request_id="acq",
            domain_handoff_reference="handoff://x",
            strategy_id="other.strategy",
        )


def test_backend_unavailable() -> None:
    class _BrokenRepo:
        def record(self, association: MarketplaceQualifiedToolStageContext):
            raise NotImplementedError

        def get_by_handoff_id(self, handoff_id: str):
            raise MarketplaceQualifiedToolStageContextAssociationUnavailableError("down")

    resolver = MarketplaceQualifiedToolStageContextResolverImpl(_BrokenRepo())
    handoff_id = derive_marketplace_gap_tool_handoff_id(tenant_id="t", operation_id="o")
    with pytest.raises(MarketplaceQualifiedToolStageContextResolverUnavailableError):
        resolver.resolve_for_qualification(
            acquisition_request_id="o",
            domain_handoff_reference=marketplace_domain_handoff_reference(handoff_id),
            strategy_id=_STRATEGY,
        )


def test_two_tenants_same_acquisition_id_independent() -> None:
    resolver, repo = _resolver()
    acquisition_id = "same-acq"
    for tenant in ("tenant-a", "tenant-b"):
        handoff_id = derive_marketplace_gap_tool_handoff_id(
            tenant_id=tenant,
            operation_id=acquisition_id,
        )
        repo.record(
            MarketplaceQualifiedToolStageContext(
                handoff_id=handoff_id,
                tenant_id=tenant,
                acquisition_request_id=acquisition_id,
            ),
        )
        ctx = resolver.resolve_for_qualification(
            acquisition_request_id=acquisition_id,
            domain_handoff_reference=marketplace_domain_handoff_reference(handoff_id),
            strategy_id=_STRATEGY,
        )
        assert ctx.tenant_id == tenant
