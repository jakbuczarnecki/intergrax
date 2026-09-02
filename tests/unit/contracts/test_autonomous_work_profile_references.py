# © Artur Czarnecki. All rights reserved.

"""AW-1B — Autonomous Work versioned profile-reference contract tests."""

from __future__ import annotations

from dataclasses import fields
from typing import get_type_hints

import pytest

from intergrax.contracts.autonomous_work import (
    ProfileVersion,
    WorkerDefinition,
    initial_profile_version,
)
from intergrax.contracts.autonomous_work.profile_reference import (
    BudgetProfileRef,
    CapabilityProfileRef,
    CodecraftProfileRef,
    CollaborationProfileRef,
    EscalationPolicyRef,
    GovernanceProfileRef,
    MemoryProfileRef,
    ObservabilityProfileRef,
    RiskProfileRef,
    ScheduleProfileRef,
)

_PROFILE_REF_TYPES = (
    GovernanceProfileRef,
    BudgetProfileRef,
    MemoryProfileRef,
    CapabilityProfileRef,
    CodecraftProfileRef,
    RiskProfileRef,
    ScheduleProfileRef,
    EscalationPolicyRef,
    CollaborationProfileRef,
    ObservabilityProfileRef,
)

_REQUIRED_WORKER_PROFILE_FIELDS = (
    "governance_profile_ref",
    "budget_profile_ref",
    "memory_profile_ref",
    "capability_profile_ref",
    "codecraft_profile_ref",
    "risk_profile_ref",
    "schedule_profile_ref",
    "escalation_policy_ref",
    "collaboration_profile_ref",
    "observability_profile_ref",
)

_FORBIDDEN_PROFILE_REF_FIELDS = (
    "credentials",
    "endpoint",
    "provider",
    "implementation",
    "backend_config",
    "memory_config",
    "policy_config",
    "secret",
    "api_key",
)

_FORBIDDEN_WORKER_PLUGIN_FIELDS = (
    "plugin",
    "plugin_name",
    "provider",
    "implementation",
    "adapter",
    "factory",
    "module_path",
    "class_name",
    "profiles",
    "plugins",
)


def _profile_ref(
    ref_type: type[MemoryProfileRef],
    *,
    profile_id: str = "memory/order-operations",
    version: ProfileVersion | None = None,
) -> MemoryProfileRef:
    return ref_type(
        profile_id=profile_id,
        version=version or initial_profile_version(),
    )


@pytest.mark.unit
@pytest.mark.parametrize("ref_type", _PROFILE_REF_TYPES)
def test_profile_ref_valid_construction(ref_type: type[MemoryProfileRef]) -> None:
    ref = _profile_ref(ref_type, profile_id=f"{ref_type.__name__}/default")
    assert ref.profile_id == f"{ref_type.__name__}/default"
    assert ref.version == initial_profile_version()


@pytest.mark.unit
@pytest.mark.parametrize("ref_type", _PROFILE_REF_TYPES)
def test_profile_ref_rejects_empty_profile_id(ref_type: type[MemoryProfileRef]) -> None:
    with pytest.raises(ValueError, match="profile_id"):
        ref_type(profile_id="", version=initial_profile_version())


@pytest.mark.unit
@pytest.mark.parametrize("ref_type", _PROFILE_REF_TYPES)
def test_profile_ref_rejects_whitespace_only_profile_id(
    ref_type: type[MemoryProfileRef],
) -> None:
    with pytest.raises(ValueError, match="profile_id"):
        ref_type(profile_id="   ", version=initial_profile_version())


@pytest.mark.unit
@pytest.mark.parametrize("ref_type", _PROFILE_REF_TYPES)
def test_profile_ref_rejects_surrounding_whitespace_profile_id(
    ref_type: type[MemoryProfileRef],
) -> None:
    with pytest.raises(ValueError, match="profile_id"):
        ref_type(profile_id=" memory/order-operations ", version=initial_profile_version())


@pytest.mark.unit
@pytest.mark.parametrize("invalid_version", [True, -1, "1", 1.0])
def test_profile_ref_rejects_invalid_version(invalid_version: object) -> None:
    with pytest.raises((TypeError, ValueError), match="ProfileVersion"):
        MemoryProfileRef(
            profile_id="memory/order-operations",
            version=ProfileVersion(invalid_version),  # type: ignore[arg-type]
        )


@pytest.mark.unit
def test_profile_ref_types_are_strongly_distinct() -> None:
    version = initial_profile_version()
    memory_ref = MemoryProfileRef(
        profile_id="memory/order-operations",
        version=version,
    )
    governance_ref = GovernanceProfileRef(
        profile_id="memory/order-operations",
        version=version,
    )
    budget_ref = BudgetProfileRef(
        profile_id="memory/order-operations",
        version=version,
    )
    assert type(memory_ref) is not type(governance_ref)
    assert type(memory_ref) is not type(budget_ref)
    assert memory_ref != governance_ref
    assert memory_ref != budget_ref


@pytest.mark.unit
def test_profile_ref_types_are_statically_distinct_in_worker_definition() -> None:
    hints = get_type_hints(WorkerDefinition)
    assert hints["memory_profile_ref"] is MemoryProfileRef
    assert hints["governance_profile_ref"] is GovernanceProfileRef
    assert hints["budget_profile_ref"] is BudgetProfileRef
    assert hints["memory_profile_ref"] is not hints["governance_profile_ref"]


@pytest.mark.unit
def test_profile_refs_are_immutable() -> None:
    ref = MemoryProfileRef(
        profile_id="memory/order-operations",
        version=initial_profile_version(),
    )
    with pytest.raises(AttributeError):
        ref.profile_id = "memory/other"  # type: ignore[misc]
    with pytest.raises(AttributeError):
        ref.version = ProfileVersion(1)  # type: ignore[misc]


@pytest.mark.unit
def test_profile_ref_equality_and_version_determinism() -> None:
    first = MemoryProfileRef(
        profile_id="memory/order-operations",
        version=ProfileVersion(1),
    )
    same = MemoryProfileRef(
        profile_id="memory/order-operations",
        version=ProfileVersion(1),
    )
    different_version = MemoryProfileRef(
        profile_id="memory/order-operations",
        version=ProfileVersion(2),
    )
    assert first == same
    assert first != different_version
    assert hash(first) == hash(same)
    assert hash(first) != hash(different_version)


@pytest.mark.unit
def test_profile_refs_are_hashable() -> None:
    refs = {
        MemoryProfileRef(
            profile_id="memory/order-operations",
            version=ProfileVersion(1),
        ),
        GovernanceProfileRef(
            profile_id="governance/default",
            version=ProfileVersion(1),
        ),
    }
    assert len(refs) == 2


@pytest.mark.unit
def test_worker_definition_has_required_typed_profile_fields() -> None:
    field_names = {field.name for field in fields(WorkerDefinition)}
    for field_name in _REQUIRED_WORKER_PROFILE_FIELDS:
        assert field_name in field_names


@pytest.mark.unit
def test_profile_refs_do_not_embed_foreign_domain_configuration() -> None:
    for ref_type in _PROFILE_REF_TYPES:
        field_names = {field.name for field in fields(ref_type)}
        assert field_names == {"profile_id", "version"}
        assert set(_FORBIDDEN_PROFILE_REF_FIELDS).isdisjoint(field_names)


@pytest.mark.unit
def test_worker_definition_has_no_plugin_or_provider_identity_fields() -> None:
    field_names = {field.name for field in fields(WorkerDefinition)}
    assert set(_FORBIDDEN_WORKER_PLUGIN_FIELDS).isdisjoint(field_names)


@pytest.mark.unit
def test_worker_definition_has_no_generic_profile_dictionary() -> None:
    field_names = {field.name for field in fields(WorkerDefinition)}
    assert "profiles" not in field_names
    assert "plugins" not in field_names
