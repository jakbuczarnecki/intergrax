# © Artur Czarnecki. All rights reserved.

"""UCA-1 — DiscoveryCompletion / Gap / RealizationNeed contract matrix."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from intergrax.contracts.capability_catalog.capability_gap import (
    CapabilityGap,
    derive_capability_gap_id,
)
from intergrax.contracts.capability_catalog.capability_realization_need import (
    CapabilityRealizationNeed,
)
from intergrax.contracts.capability_catalog.discovery_completion import (
    DiscoveryCompletionOutcome,
    build_discovery_completion,
)
from intergrax.contracts.capability_catalog.federation import (
    CapabilityCatalogFederationCompleteness,
)
from intergrax.contracts.capability_catalog.identity_key import CapabilityIdentityKey
from intergrax.contracts.capability_catalog.identity import CapabilitySourceKind
from intergrax.contracts.capability_catalog.kind import CapabilityKind

pytestmark = pytest.mark.unit

_CREATED_AT = datetime(2026, 9, 20, 8, 0, tzinfo=UTC)


def _key(logical_id: str = "tools.alpha") -> CapabilityIdentityKey:
    return CapabilityIdentityKey(
        kind=CapabilityKind.TOOL,
        source_id="official.catalog",
        source_kind=CapabilitySourceKind.OFFICIAL,
        logical_id=logical_id,
    )


def test_host_available_allowed_suitable_is_direct_reuse() -> None:
    completion = build_discovery_completion(
        need_id="need-1",
        discovery_correlation_id="corr-1",
        federation_completeness=CapabilityCatalogFederationCompleteness.COMPLETE,
        suitable_host_allowed_keys=(_key(),),
        created_at=_CREATED_AT,
    )
    assert completion.outcome is DiscoveryCompletionOutcome.DIRECT_REUSE


def test_catalog_available_allowed_suitable_is_realization_required() -> None:
    completion = build_discovery_completion(
        need_id="need-1",
        discovery_correlation_id="corr-1",
        federation_completeness=CapabilityCatalogFederationCompleteness.COMPLETE,
        suitable_catalog_allowed_keys=(_key(),),
        created_at=_CREATED_AT,
    )
    assert completion.outcome is DiscoveryCompletionOutcome.REALIZATION_REQUIRED
    realization = CapabilityRealizationNeed.from_discovery_completion(completion)
    assert realization.capability_identity == _key()
    assert realization.availability.value == "catalog_available"
    assert realization.governance_disposition.value == "allowed"


def test_complete_no_suitable_allows_gap() -> None:
    completion = build_discovery_completion(
        need_id="need-1",
        discovery_correlation_id="corr-1",
        federation_completeness=CapabilityCatalogFederationCompleteness.COMPLETE,
        created_at=_CREATED_AT,
    )
    assert completion.outcome is DiscoveryCompletionOutcome.MISSING_CAPABILITY
    gap = CapabilityGap.from_discovery_completion(completion)
    assert gap.gap_id == derive_capability_gap_id(
        need_id="need-1",
        discovery_correlation_id="corr-1",
    )


def test_partial_no_suitable_is_incomplete_and_forbids_gap() -> None:
    completion = build_discovery_completion(
        need_id="need-1",
        discovery_correlation_id="corr-1",
        federation_completeness=CapabilityCatalogFederationCompleteness.PARTIAL,
        created_at=_CREATED_AT,
    )
    assert completion.outcome is DiscoveryCompletionOutcome.INCOMPLETE
    with pytest.raises(ValueError, match="MISSING_CAPABILITY"):
        CapabilityGap.from_discovery_completion(completion)


def test_blocked_forbids_gap() -> None:
    completion = build_discovery_completion(
        need_id="need-1",
        discovery_correlation_id="corr-1",
        federation_completeness=CapabilityCatalogFederationCompleteness.COMPLETE,
        governance_blocked=True,
        created_at=_CREATED_AT,
    )
    assert completion.outcome is DiscoveryCompletionOutcome.BLOCKED
    with pytest.raises(ValueError, match="MISSING_CAPABILITY"):
        CapabilityGap.from_discovery_completion(completion)


def test_scope_unavailable_forbids_gap_and_is_not_blocked() -> None:
    completion = build_discovery_completion(
        need_id="need-1",
        discovery_correlation_id="corr-1",
        federation_completeness=CapabilityCatalogFederationCompleteness.COMPLETE,
        scope_unavailable=True,
        created_at=_CREATED_AT,
    )
    assert completion.outcome is DiscoveryCompletionOutcome.SCOPE_UNAVAILABLE
    assert completion.outcome is not DiscoveryCompletionOutcome.BLOCKED
    with pytest.raises(ValueError, match="MISSING_CAPABILITY"):
        CapabilityGap.from_discovery_completion(completion)


def test_unavailable_forbids_gap() -> None:
    completion = build_discovery_completion(
        need_id="need-1",
        discovery_correlation_id="corr-1",
        federation_completeness=CapabilityCatalogFederationCompleteness.COMPLETE,
        unavailable=True,
        created_at=_CREATED_AT,
    )
    assert completion.outcome is DiscoveryCompletionOutcome.UNAVAILABLE
    with pytest.raises(ValueError, match="MISSING_CAPABILITY"):
        CapabilityGap.from_discovery_completion(completion)


def test_conflict_forbids_gap() -> None:
    completion = build_discovery_completion(
        need_id="need-1",
        discovery_correlation_id="corr-1",
        federation_completeness=CapabilityCatalogFederationCompleteness.COMPLETE,
        conflict=True,
        created_at=_CREATED_AT,
    )
    assert completion.outcome is DiscoveryCompletionOutcome.CONFLICT
    with pytest.raises(ValueError, match="MISSING_CAPABILITY"):
        CapabilityGap.from_discovery_completion(completion)


def test_gap_forbids_catalog_available_candidate() -> None:
    completion = build_discovery_completion(
        need_id="need-1",
        discovery_correlation_id="corr-1",
        federation_completeness=CapabilityCatalogFederationCompleteness.COMPLETE,
        suitable_catalog_allowed_keys=(_key(),),
        created_at=_CREATED_AT,
    )
    with pytest.raises(ValueError, match="MISSING_CAPABILITY"):
        CapabilityGap.from_discovery_completion(completion)


def test_gap_forbids_host_available_candidate() -> None:
    completion = build_discovery_completion(
        need_id="need-1",
        discovery_correlation_id="corr-1",
        federation_completeness=CapabilityCatalogFederationCompleteness.COMPLETE,
        suitable_host_allowed_keys=(_key(),),
        created_at=_CREATED_AT,
    )
    with pytest.raises(ValueError, match="MISSING_CAPABILITY"):
        CapabilityGap.from_discovery_completion(completion)


def test_realization_forbids_non_realization_outcome() -> None:
    completion = build_discovery_completion(
        need_id="need-1",
        discovery_correlation_id="corr-1",
        federation_completeness=CapabilityCatalogFederationCompleteness.COMPLETE,
        suitable_host_allowed_keys=(_key(),),
        created_at=_CREATED_AT,
    )
    with pytest.raises(ValueError, match="REALIZATION_REQUIRED"):
        CapabilityRealizationNeed.from_discovery_completion(completion)


def test_outcome_mismatch_is_rejected() -> None:
    from intergrax.contracts.capability_catalog.discovery_completion import (
        DiscoveryCompletion,
    )

    with pytest.raises(ValidationError):
        DiscoveryCompletion(
            need_id="need-1",
            discovery_correlation_id="corr-1",
            federation_completeness=CapabilityCatalogFederationCompleteness.COMPLETE,
            suitable_host_allowed_keys=(_key(),),
            outcome=DiscoveryCompletionOutcome.MISSING_CAPABILITY,
            created_at=_CREATED_AT,
        )


def test_gap_id_is_deterministic() -> None:
    first = derive_capability_gap_id(need_id="need-1", discovery_correlation_id="corr-1")
    second = derive_capability_gap_id(need_id="need-1", discovery_correlation_id="corr-1")
    assert first == second
    assert first == "capability-gap:need-1:corr-1"
