# © Artur Czarnecki. All rights reserved.

"""TRACE-X-P5-P0 mechanical qualification gates."""

from __future__ import annotations

import subprocess
from dataclasses import replace

import pytest

from intergrax.contracts.control_plane_mutation import (
    control_plane_mutation_request_digest,
    evidence_from_request_and_decision,
)
from intergrax.contracts.runtime_policy import PolicyAction, PolicyDecision
from intergrax.integrations.contracts.existing_capability_configuration import (
    ExistingCapabilityConfigurationRealizationError,
    ExistingCapabilityConfigurationRealizationFailureReason,
)
from tests.qualification.trace_x._trace_x_p5_support import (
    CONFIGURATION_PROVENANCE_SURFACE_REGISTRY,
    FRZ_TRC_P5_DISPOSITION,
    POLICY_PROVENANCE_SURFACE_REGISTRY,
    PROFILE_REVISION_SURFACE_REGISTRY,
    PROVENANCE_GAPS,
    PROVENANCE_JOINS,
    TRACE_X_P5_P0_START_HEAD,
    compare_configuration_surfaces_to_registry,
    compare_policy_surfaces_to_registry,
    compare_profile_surfaces_to_registry,
    discover_configuration_provenance_surfaces,
    discover_policy_provenance_surfaces,
    discover_profile_revision_surfaces,
)
from tests.qualification.trace_x._trace_x_p4_registry_types import compare_discovered_to_registry

pytestmark = [pytest.mark.qualification, pytest.mark.gate]


def test_txp5p0_q01_start_head_ancestry() -> None:
    subprocess.check_call(
        ["git", "merge-base", "--is-ancestor", TRACE_X_P5_P0_START_HEAD, "HEAD"],
    )


def test_txp5p0_q02_policy_discovery_registry_parity() -> None:
    discovered = discover_policy_provenance_surfaces()
    result = compare_policy_surfaces_to_registry(discovered)
    assert not result.unknown, f"unclassified policy surfaces: {sorted(result.unknown)}"
    assert not result.orphan, f"orphan policy registry rows: {sorted(result.orphan)}"
    assert not result.duplicate_registry_keys
    assert len(discovered) == len(POLICY_PROVENANCE_SURFACE_REGISTRY)


def test_txp5p0_q03_profile_discovery_registry_parity() -> None:
    discovered = discover_profile_revision_surfaces()
    result = compare_profile_surfaces_to_registry(discovered)
    assert not result.unknown, f"unclassified profile surfaces: {sorted(result.unknown)}"
    assert not result.orphan, f"orphan profile registry rows: {sorted(result.orphan)}"
    assert not result.duplicate_registry_keys
    assert len(discovered) == len(PROFILE_REVISION_SURFACE_REGISTRY)


def test_txp5p0_q04_config_discovery_registry_parity() -> None:
    discovered = discover_configuration_provenance_surfaces()
    result = compare_configuration_surfaces_to_registry(discovered)
    assert not result.unknown, f"unclassified configuration surfaces: {sorted(result.unknown)}"
    assert not result.orphan, f"orphan configuration registry rows: {sorted(result.orphan)}"
    assert not result.duplicate_registry_keys
    assert len(discovered) == len(CONFIGURATION_PROVENANCE_SURFACE_REGISTRY)


def test_txp5p0_q05_provenance_joins_non_heuristic() -> None:
    assert PROVENANCE_JOINS
    assert all(not join.heuristic for join in PROVENANCE_JOINS)
    assert len({join.join_id for join in PROVENANCE_JOINS}) == len(PROVENANCE_JOINS)


def test_txp5p0_q06_synthetic_policy_surface_negative() -> None:
    discovered = discover_policy_provenance_surfaces()
    injected = discovered | {("intergrax/synthetic/policy_probe.py", "SyntheticPolicySurface")}
    result = compare_policy_surfaces_to_registry(injected)
    assert "intergrax/synthetic/policy_probe.py" in {k[0] for k in result.unknown}


def test_txp5p0_q07_synthetic_profile_surface_negative() -> None:
    discovered = discover_profile_revision_surfaces()
    injected = discovered | {("intergrax/synthetic/profile_probe.py", "SyntheticProfileSurface")}
    result = compare_profile_surfaces_to_registry(injected)
    assert "intergrax/synthetic/profile_probe.py" in {k[0] for k in result.unknown}


def test_txp5p0_q08_synthetic_configuration_surface_negative() -> None:
    discovered = discover_configuration_provenance_surfaces()
    injected = discovered | {
        ("intergrax/synthetic/config_probe.py", "SyntheticConfigurationSurface"),
    }
    result = compare_configuration_surfaces_to_registry(injected)
    assert "intergrax/synthetic/config_probe.py" in {k[0] for k in result.unknown}


def test_txp5p0_q09_configured_fingerprint_mismatch_fail_closed() -> None:
    from tests.unit.integrations.test_existing_capability_configuration_service import (
        _payload,
        _principal,
        _request,
    )

    request = _request(configuration_fingerprint="fp-a")
    with pytest.raises(ExistingCapabilityConfigurationRealizationError) as exc:
        replace(request, configuration=_payload("fp-b"))
    assert exc.value.reason is ExistingCapabilityConfigurationRealizationFailureReason.IDENTITY_MISMATCH


def test_txp5p0_q10_configuration_tenant_mismatch_fail_closed() -> None:
    from intergrax.integrations.contracts.existing_capability_configuration import (
        verify_admitted_authorization_evidence,
        project_control_plane_mutation_request,
    )
    from tests.unit.integrations.test_existing_capability_configuration_service import (
        _payload,
        _principal,
        _request,
    )

    request = _request(tenant_id="tenant-a", principal=_principal(tenant_id="tenant-a"))
    governance_request = project_control_plane_mutation_request(request)
    digest = control_plane_mutation_request_digest(governance_request)
    evidence = evidence_from_request_and_decision(
        governance_request,
        decision=PolicyDecision(action=PolicyAction.ALLOW),
        request_digest=digest,
    ).model_copy(update={"tenant_id": "tenant-b"})
    with pytest.raises(ExistingCapabilityConfigurationRealizationError) as exc:
        verify_admitted_authorization_evidence(
            request=request,
            governance_request=governance_request,
            evidence=evidence,
        )
    assert exc.value.reason is ExistingCapabilityConfigurationRealizationFailureReason.TENANT_MISMATCH


def test_txp5p0_q11_profile_binding_tenant_mismatch_fail_closed() -> None:
    from intergrax.contracts.execution_identity import mint_execution_id

    from tests.unit.applications.test_effective_profile_revision import (
        _application,
        _revision_from_layers,
    )
    from intergrax.applications._shared.profile_resolution.execution_pinning import (
        InMemoryEffectiveProfileExecutionPinningStore,
        pin_effective_profile_revision_for_execution,
    )
    from intergrax.applications._shared.profile_resolution.store import (
        InMemoryEffectiveProfileRevisionStore,
    )

    revision_store = InMemoryEffectiveProfileRevisionStore()
    pinning_store = InMemoryEffectiveProfileExecutionPinningStore()
    _, revision = _revision_from_layers(_application(), (), store=revision_store)
    execution_id = mint_execution_id()
    pin_effective_profile_revision_for_execution(
        revision=revision,
        tenant_id="tenant-a",
        execution_id=execution_id,
        pinning_store=pinning_store,
        revision_store=revision_store,
    )
    assert pinning_store.get(tenant_id="tenant-b", execution_id=execution_id) is None


def test_txp5p0_q12_frz_dispositions_not_pass() -> None:
    from tests.qualification.trace_x._trace_x_p5_registry_types import ProvenanceDisposition

    assert set(FRZ_TRC_P5_DISPOSITION) == {"FRZ-TRC-07", "FRZ-TRC-08", "FRZ-TRC-11"}
    assert all(
        disposition == ProvenanceDisposition.PARTIAL_CURRENT_HEAD
        for disposition in FRZ_TRC_P5_DISPOSITION.values()
    )
    blockers = [g for g in PROVENANCE_GAPS if g.classification == "IN-SCOPE BLOCKER"]
    assert blockers


def test_txp5p0_q13_registry_duplicate_owner_gate() -> None:
    for registry in (
        POLICY_PROVENANCE_SURFACE_REGISTRY,
        PROFILE_REVISION_SURFACE_REGISTRY,
        CONFIGURATION_PROVENANCE_SURFACE_REGISTRY,
    ):
        result = compare_discovered_to_registry(
            frozenset(row.key for row in registry),
            registry,
        )
        assert not result.duplicate_registry_keys
