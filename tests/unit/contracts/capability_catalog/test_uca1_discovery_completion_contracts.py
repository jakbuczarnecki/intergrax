# © Artur Czarnecki. All rights reserved.

"""UCA-1R / UCA-1R2 — DiscoveryCompletion precedence & fact consistency."""

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
    derive_discovery_completion_outcome,
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


# --- baseline outcomes ---


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
    selected = _key()
    completion = build_discovery_completion(
        need_id="need-1",
        discovery_correlation_id="corr-1",
        federation_completeness=CapabilityCatalogFederationCompleteness.COMPLETE,
        suitable_catalog_allowed_keys=(selected,),
        created_at=_CREATED_AT,
    )
    assert completion.outcome is DiscoveryCompletionOutcome.REALIZATION_REQUIRED
    realization = CapabilityRealizationNeed.from_discovery_completion(
        completion,
        capability_identity=selected,
    )
    assert realization.capability_identity == selected
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


# --- explicit realization selection ---


def test_realization_single_catalog_candidate_with_explicit_identity() -> None:
    selected = _key("tools.alpha")
    completion = build_discovery_completion(
        need_id="need-1",
        discovery_correlation_id="corr-1",
        federation_completeness=CapabilityCatalogFederationCompleteness.COMPLETE,
        suitable_catalog_allowed_keys=(selected,),
        created_at=_CREATED_AT,
    )
    realization = CapabilityRealizationNeed.from_discovery_completion(
        completion,
        capability_identity=selected,
    )
    assert realization.capability_identity == selected


def test_realization_multiple_catalog_candidates_uses_exact_selected_identity() -> None:
    alpha = _key("tools.alpha")
    beta = _key("tools.beta")
    completion = build_discovery_completion(
        need_id="need-1",
        discovery_correlation_id="corr-1",
        federation_completeness=CapabilityCatalogFederationCompleteness.COMPLETE,
        suitable_catalog_allowed_keys=(alpha, beta),
        created_at=_CREATED_AT,
    )
    realization = CapabilityRealizationNeed.from_discovery_completion(
        completion,
        capability_identity=beta,
    )
    assert realization.capability_identity == beta
    assert realization.capability_identity != alpha


def test_realization_omitted_identity_is_api_error() -> None:
    completion = build_discovery_completion(
        need_id="need-1",
        discovery_correlation_id="corr-1",
        federation_completeness=CapabilityCatalogFederationCompleteness.COMPLETE,
        suitable_catalog_allowed_keys=(_key("tools.alpha"), _key("tools.beta")),
        created_at=_CREATED_AT,
    )
    with pytest.raises(TypeError):
        CapabilityRealizationNeed.from_discovery_completion(completion)  # type: ignore[call-arg]


def test_realization_identity_outside_catalog_keys_errors() -> None:
    completion = build_discovery_completion(
        need_id="need-1",
        discovery_correlation_id="corr-1",
        federation_completeness=CapabilityCatalogFederationCompleteness.COMPLETE,
        suitable_catalog_allowed_keys=(_key("tools.alpha"),),
        created_at=_CREATED_AT,
    )
    with pytest.raises(ValueError, match="suitable_catalog_allowed_keys"):
        CapabilityRealizationNeed.from_discovery_completion(
            completion,
            capability_identity=_key("tools.other"),
        )


def test_realization_host_available_identity_errors() -> None:
    host = _key("tools.host")
    catalog = _key("tools.catalog")
    # HOST alone yields DIRECT_REUSE — cannot build realization from that outcome.
    host_only = build_discovery_completion(
        need_id="need-1",
        discovery_correlation_id="corr-1",
        federation_completeness=CapabilityCatalogFederationCompleteness.COMPLETE,
        suitable_host_allowed_keys=(host,),
        created_at=_CREATED_AT,
    )
    with pytest.raises(ValueError, match="REALIZATION_REQUIRED"):
        CapabilityRealizationNeed.from_discovery_completion(
            host_only,
            capability_identity=host,
        )
    # Catalog realization must reject a host identity that is not in catalog keys.
    catalog_completion = build_discovery_completion(
        need_id="need-1",
        discovery_correlation_id="corr-1",
        federation_completeness=CapabilityCatalogFederationCompleteness.COMPLETE,
        suitable_catalog_allowed_keys=(catalog,),
        created_at=_CREATED_AT,
    )
    with pytest.raises(ValueError, match="suitable_catalog_allowed_keys"):
        CapabilityRealizationNeed.from_discovery_completion(
            catalog_completion,
            capability_identity=host,
        )


# --- PARTIAL semantics ---


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


def test_partial_plus_host_candidate_is_direct_reuse() -> None:
    completion = build_discovery_completion(
        need_id="need-1",
        discovery_correlation_id="corr-1",
        federation_completeness=CapabilityCatalogFederationCompleteness.PARTIAL,
        suitable_host_allowed_keys=(_key(),),
        created_at=_CREATED_AT,
    )
    assert completion.outcome is DiscoveryCompletionOutcome.DIRECT_REUSE


def test_partial_plus_catalog_candidate_is_realization_required() -> None:
    selected = _key()
    completion = build_discovery_completion(
        need_id="need-1",
        discovery_correlation_id="corr-1",
        federation_completeness=CapabilityCatalogFederationCompleteness.PARTIAL,
        suitable_catalog_allowed_keys=(selected,),
        created_at=_CREATED_AT,
    )
    assert completion.outcome is DiscoveryCompletionOutcome.REALIZATION_REQUIRED
    realization = CapabilityRealizationNeed.from_discovery_completion(
        completion,
        capability_identity=selected,
    )
    assert realization.capability_identity == selected


def test_complete_required_to_prove_absence_not_for_positive_candidate() -> None:
    """COMPLETE proves absence; positive candidates do not require COMPLETE."""
    assert (
        derive_discovery_completion_outcome(
            federation_completeness=CapabilityCatalogFederationCompleteness.PARTIAL,
            suitable_host_allowed_keys=(),
            suitable_catalog_allowed_keys=(),
            governance_blocked=False,
            availability_blocked=False,
            scope_unavailable=False,
            unavailable=False,
            conflict=False,
        )
        is DiscoveryCompletionOutcome.INCOMPLETE
    )
    assert (
        derive_discovery_completion_outcome(
            federation_completeness=CapabilityCatalogFederationCompleteness.PARTIAL,
            suitable_host_allowed_keys=(_key(),),
            suitable_catalog_allowed_keys=(),
            governance_blocked=False,
            availability_blocked=False,
            scope_unavailable=False,
            unavailable=False,
            conflict=False,
        )
        is DiscoveryCompletionOutcome.DIRECT_REUSE
    )
    assert (
        derive_discovery_completion_outcome(
            federation_completeness=CapabilityCatalogFederationCompleteness.COMPLETE,
            suitable_host_allowed_keys=(),
            suitable_catalog_allowed_keys=(),
            governance_blocked=False,
            availability_blocked=False,
            scope_unavailable=False,
            unavailable=False,
            conflict=False,
        )
        is DiscoveryCompletionOutcome.MISSING_CAPABILITY
    )


# --- result-level blocker + positive candidate (unrepresentable) ---


def test_conflict_plus_candidate_is_conflict_fail_closed() -> None:
    completion = build_discovery_completion(
        need_id="need-1",
        discovery_correlation_id="corr-1",
        federation_completeness=CapabilityCatalogFederationCompleteness.COMPLETE,
        suitable_catalog_allowed_keys=(_key(),),
        conflict=True,
        created_at=_CREATED_AT,
    )
    assert completion.outcome is DiscoveryCompletionOutcome.CONFLICT
    with pytest.raises(ValueError, match="REALIZATION_REQUIRED"):
        CapabilityRealizationNeed.from_discovery_completion(
            completion,
            capability_identity=_key(),
        )
    with pytest.raises(ValueError, match="MISSING_CAPABILITY"):
        CapabilityGap.from_discovery_completion(completion)


def test_conflict_alone_is_conflict() -> None:
    completion = build_discovery_completion(
        need_id="need-1",
        discovery_correlation_id="corr-1",
        federation_completeness=CapabilityCatalogFederationCompleteness.COMPLETE,
        conflict=True,
        created_at=_CREATED_AT,
    )
    assert completion.outcome is DiscoveryCompletionOutcome.CONFLICT


def test_catalog_candidate_plus_governance_blocked_is_invalid() -> None:
    with pytest.raises(ValueError, match="result-level scope/unavailable/blocked"):
        build_discovery_completion(
            need_id="need-1",
            discovery_correlation_id="corr-1",
            federation_completeness=CapabilityCatalogFederationCompleteness.COMPLETE,
            suitable_catalog_allowed_keys=(_key(),),
            governance_blocked=True,
            created_at=_CREATED_AT,
        )


def test_catalog_candidate_plus_availability_blocked_is_invalid() -> None:
    with pytest.raises(ValueError, match="result-level scope/unavailable/blocked"):
        build_discovery_completion(
            need_id="need-1",
            discovery_correlation_id="corr-1",
            federation_completeness=CapabilityCatalogFederationCompleteness.COMPLETE,
            suitable_catalog_allowed_keys=(_key(),),
            availability_blocked=True,
            created_at=_CREATED_AT,
        )


def test_catalog_candidate_plus_unavailable_is_invalid() -> None:
    with pytest.raises(ValueError, match="result-level scope/unavailable/blocked"):
        build_discovery_completion(
            need_id="need-1",
            discovery_correlation_id="corr-1",
            federation_completeness=CapabilityCatalogFederationCompleteness.COMPLETE,
            suitable_catalog_allowed_keys=(_key(),),
            unavailable=True,
            created_at=_CREATED_AT,
        )


def test_host_candidate_plus_scope_unavailable_is_invalid() -> None:
    with pytest.raises(ValueError, match="result-level scope/unavailable/blocked"):
        build_discovery_completion(
            need_id="need-1",
            discovery_correlation_id="corr-1",
            federation_completeness=CapabilityCatalogFederationCompleteness.COMPLETE,
            suitable_host_allowed_keys=(_key(),),
            scope_unavailable=True,
            created_at=_CREATED_AT,
        )


def test_catalog_candidate_plus_scope_unavailable_is_invalid() -> None:
    with pytest.raises(ValueError, match="result-level scope/unavailable/blocked"):
        build_discovery_completion(
            need_id="need-1",
            discovery_correlation_id="corr-1",
            federation_completeness=CapabilityCatalogFederationCompleteness.COMPLETE,
            suitable_catalog_allowed_keys=(_key(),),
            scope_unavailable=True,
            created_at=_CREATED_AT,
        )


def test_catalog_candidate_plus_multiple_result_blockers_is_invalid() -> None:
    with pytest.raises(ValueError, match="result-level scope/unavailable/blocked"):
        build_discovery_completion(
            need_id="need-1",
            discovery_correlation_id="corr-1",
            federation_completeness=CapabilityCatalogFederationCompleteness.COMPLETE,
            suitable_catalog_allowed_keys=(_key(),),
            governance_blocked=True,
            availability_blocked=True,
            unavailable=True,
            created_at=_CREATED_AT,
        )


def test_model_rejects_positive_candidate_with_result_level_blocker() -> None:
    """Direct model construction surfaces the same inconsistency as ValidationError."""
    from intergrax.contracts.capability_catalog.discovery_completion import (
        DiscoveryCompletion,
    )

    with pytest.raises(ValidationError, match="result-level scope/unavailable/blocked"):
        DiscoveryCompletion(
            need_id="need-1",
            discovery_correlation_id="corr-1",
            federation_completeness=CapabilityCatalogFederationCompleteness.COMPLETE,
            suitable_catalog_allowed_keys=(_key(),),
            governance_blocked=True,
            outcome=DiscoveryCompletionOutcome.REALIZATION_REQUIRED,
            created_at=_CREATED_AT,
        )


def test_derive_rejects_positive_candidate_with_result_level_blocker() -> None:
    with pytest.raises(ValueError, match="result-level scope/unavailable/blocked"):
        derive_discovery_completion_outcome(
            federation_completeness=CapabilityCatalogFederationCompleteness.COMPLETE,
            suitable_host_allowed_keys=(_key(),),
            suitable_catalog_allowed_keys=(),
            governance_blocked=False,
            availability_blocked=False,
            scope_unavailable=True,
            unavailable=False,
            conflict=False,
        )


def test_scope_unavailable_without_candidate_is_scope_unavailable() -> None:
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


def test_unavailable_without_candidate_forbids_gap() -> None:
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


def test_blocked_without_candidate_forbids_gap() -> None:
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
        CapabilityRealizationNeed.from_discovery_completion(
            completion,
            capability_identity=_key(),
        )


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


def test_host_and_catalog_overlap_rejected() -> None:
    with pytest.raises(ValidationError, match="disjoint"):
        build_discovery_completion(
            need_id="need-1",
            discovery_correlation_id="corr-1",
            federation_completeness=CapabilityCatalogFederationCompleteness.COMPLETE,
            suitable_host_allowed_keys=(_key("tools.same"),),
            suitable_catalog_allowed_keys=(_key("tools.same"),),
            created_at=_CREATED_AT,
        )


def test_gap_id_is_deterministic() -> None:
    first = derive_capability_gap_id(
        need_id="need-1", discovery_correlation_id="corr-1"
    )
    second = derive_capability_gap_id(
        need_id="need-1", discovery_correlation_id="corr-1"
    )
    assert first == second
    assert first == "capability-gap:need-1:corr-1"
