# © Artur Czarnecki. All rights reserved.

"""GR-12-A4-R2-R0 — vector identity, configuration projection, TOCTOU qualification gates."""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import pytest

from intergrax.integrations.contracts.vector_index_administration import VectorIndexIdentity
from tests.qualification.governance.gr12.catalog import (
    GR12_A4_R2_VECTOR_ADR_PATH,
    GR12_CONTROL_PLANE_SURFACES,
    Gr12Applicability,
    Gr12CoverageStatus,
)
from tests.qualification.governance.gr12.gr12_a4_r2_r0_vector_revision_semantics import (
    GR12_VECTOR_CONFIGURATION_DIGEST_EXCLUDED_RUNTIME_FIELDS,
    GR12_VECTOR_CONFIGURATION_DIGEST_INCLUDED_FIELDS,
    GR12_VECTOR_CONFIGURATION_PROJECTION_SCHEMA,
    Gr12VectorAbsentIndexRevisionState,
    Gr12VectorConfigurationDigestField,
    Gr12VectorIdentityIntrinsicValidation,
    Gr12VectorLiveOperatorIdentityValidation,
    Gr12VectorLivePrepareAuthorizationTiming,
    Gr12VectorProviderCasAvailability,
    Gr12VectorStaleAuthorizationBehavior,
    VectorIndexConfigurationProjection,
)
from tests.qualification.governance.gr12.gr12_a4_r2_vector_architecture_decision import (
    GR12_A4_R2_VECTOR_ARCHITECTURE_DECISION,
    Gr12VectorArchitecturePhase,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]


def _vector_row():
    return next(row for row in GR12_CONTROL_PLANE_SURFACES if row.path_id == "CP-VECTOR-INDEX-ADMIN")


def test_gr12_a4_r2_r0_identity_dataclass_does_not_intrinsically_validate_non_empty() -> None:
    """R0-1 — ``VectorIndexIdentity`` alone does not guarantee non-empty fields."""
    decision = GR12_A4_R2_VECTOR_ARCHITECTURE_DECISION
    assert (
        decision.identity_intrinsic_validation
        is Gr12VectorIdentityIntrinsicValidation.NONE
    )
    empty_identity = VectorIndexIdentity(logical_name="", tenant_id="")
    assert empty_identity.logical_name == ""
    assert empty_identity.tenant_id == ""


def test_gr12_a4_r2_r0_live_operator_identity_validation_required() -> None:
    """R0-2 — live operator boundary must validate before CLA-04 mapping."""
    decision = GR12_A4_R2_VECTOR_ARCHITECTURE_DECISION
    assert (
        decision.live_operator_identity_validation
        is Gr12VectorLiveOperatorIdentityValidation.REQUIRED_NON_EMPTY_LOGICAL_NAME_AND_TENANT_ID
    )


def test_gr12_a4_r2_r0_revision_projection_excludes_runtime_fields() -> None:
    """R0-3 — runtime diagnostics are not configuration authority."""
    decision = GR12_A4_R2_VECTOR_ARCHITECTURE_DECISION
    excluded = set(decision.excluded_revision_fields)
    assert "point_count" in excluded
    assert "reachable" in excluded
    for field in GR12_VECTOR_CONFIGURATION_DIGEST_EXCLUDED_RUNTIME_FIELDS:
        assert field in excluded


def test_gr12_a4_r2_r0_single_logical_projection_schema_for_current_and_target() -> None:
    """R0-4 — one schema name for both current and target digests."""
    decision = GR12_A4_R2_VECTOR_ARCHITECTURE_DECISION
    assert decision.configuration_projection_schema == GR12_VECTOR_CONFIGURATION_PROJECTION_SCHEMA
    assert decision.configuration_projection_schema == VectorIndexConfigurationProjection.__name__
    projection_fields = {item.name for item in fields(VectorIndexConfigurationProjection)}
    for digest_field in Gr12VectorConfigurationDigestField:
        assert digest_field.value in projection_fields
    assert set(decision.configuration_digest_included_fields) == {
        item.value for item in GR12_VECTOR_CONFIGURATION_DIGEST_INCLUDED_FIELDS
    }


def test_gr12_a4_r2_r0_stale_digest_invalidates_prior_authorization() -> None:
    """R0-5 — authorized digest A vs re-read B invalidates prior authorization."""
    decision = GR12_A4_R2_VECTOR_ARCHITECTURE_DECISION
    assert (
        decision.stale_authorization_behavior
        is Gr12VectorStaleAuthorizationBehavior.STALE_INVALIDATES_PRIOR_AUTHORIZATION_ABORT
    )
    assert "invalidates prior authorization" in decision.toctou_strategy.lower()


def test_gr12_a4_r2_r0_fresh_authorization_or_abort_required() -> None:
    """R0-6 — bounded fresh CLA-04 or abort; no continue without authorize."""
    decision = GR12_A4_R2_VECTOR_ARCHITECTURE_DECISION
    assert decision.stale_retry_fresh_authorization_policy.value.endswith("OR_ABORT")
    assert decision.provider_cas_available is Gr12VectorProviderCasAvailability.UNAVAILABLE


def test_gr12_a4_r2_r0_provider_cas_unavailable() -> None:
    """R0-7 — neutral port has no CAS."""
    decision = GR12_A4_R2_VECTOR_ARCHITECTURE_DECISION
    assert decision.provider_cas_available is Gr12VectorProviderCasAvailability.UNAVAILABLE


def test_gr12_a4_r2_r0_vector_catalog_still_implementation_required() -> None:
    """R0-8 — architecture reconciled; runtime not qualified."""
    row = _vector_row()
    assert row.applicability is Gr12Applicability.APPLICABLE
    assert row.coverage is Gr12CoverageStatus.IMPLEMENTATION_REQUIRED
    decision = GR12_A4_R2_VECTOR_ARCHITECTURE_DECISION
    assert decision.architecture_phase is Gr12VectorArchitecturePhase.ARCHITECTURE_DECISION_RECONCILED


def test_gr12_a4_r2_r0_live_prepare_authorization_before_invocation() -> None:
    decision = GR12_A4_R2_VECTOR_ARCHITECTURE_DECISION
    assert (
        decision.live_prepare_authorization_timing
        is Gr12VectorLivePrepareAuthorizationTiming.BEFORE_PREPARE_INDEX_INVOCATION
    )


def test_gr12_a4_r2_r0_absent_index_revision_state() -> None:
    decision = GR12_A4_R2_VECTOR_ARCHITECTURE_DECISION
    assert decision.absent_state_semantics is Gr12VectorAbsentIndexRevisionState.ABSENT


def test_gr12_a4_r2_r0_adr_reconciliation_sections_present() -> None:
    adr = (_REPO_ROOT / GR12_A4_R2_VECTOR_ADR_PATH).read_text(encoding="utf-8")
    required_sections = (
        "Identity validation",
        "Canonical configuration projection",
        "Absent index state",
        "Revision digest semantics",
        "TOCTOU invalidation",
        "No CAS guarantee",
        "Live prepare authorization timing",
        "Non-goals",
    )
    for heading in required_sections:
        assert heading in adr
