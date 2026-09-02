# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Versioned logical profile references for Autonomous Work (AW-1B)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, TypeVar

from intergrax.contracts.autonomous_work._validation import (
    require_non_empty_text,
    require_non_negative_int,
)

_TProfileRef = TypeVar("_TProfileRef", bound="_VersionedProfileReference")


def validate_profile_id(value: object) -> str:
    return require_non_empty_text(value, label="profile_id")


def validate_profile_version(value: object) -> int:
    if isinstance(value, ProfileVersion):
        return value.value
    return require_non_negative_int(value, label="ProfileVersion")


@dataclass(frozen=True, slots=True)
class ProfileVersion:
    """Immutable version of domain-owned profile configuration."""

    value: int

    def __post_init__(self) -> None:
        validate_profile_version(self.value)

    def __lt__(self, other: object) -> bool:
        if type(other) is not ProfileVersion:
            return NotImplemented
        return self.value < other.value

    def __le__(self, other: object) -> bool:
        if type(other) is not ProfileVersion:
            return NotImplemented
        return self.value <= other.value

    def __gt__(self, other: object) -> bool:
        if type(other) is not ProfileVersion:
            return NotImplemented
        return self.value > other.value

    def __ge__(self, other: object) -> bool:
        if type(other) is not ProfileVersion:
            return NotImplemented
        return self.value >= other.value


def initial_profile_version() -> ProfileVersion:
    return ProfileVersion(0)


@dataclass(frozen=True, slots=True)
class _VersionedProfileReference:
    """Shared immutable logical profile identity (profile_id + version)."""

    profile_id: str
    version: ProfileVersion

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "profile_id",
            validate_profile_id(self.profile_id),
        )
        if type(self.version) is not ProfileVersion:
            raise TypeError("version must be ProfileVersion")
        validate_profile_version(self.version)


@dataclass(frozen=True, slots=True)
class GovernanceProfileRef(_VersionedProfileReference):
    """Logical versioned reference to domain-owned Governance configuration."""


@dataclass(frozen=True, slots=True)
class BudgetProfileRef(_VersionedProfileReference):
    """Logical versioned reference to domain-owned Budget configuration."""


@dataclass(frozen=True, slots=True)
class MemoryProfileRef(_VersionedProfileReference):
    """Logical versioned reference to domain-owned Memory configuration."""


@dataclass(frozen=True, slots=True)
class CapabilityProfileRef(_VersionedProfileReference):
    """Logical versioned reference to domain-owned Capability configuration."""


@dataclass(frozen=True, slots=True)
class CodecraftProfileRef(_VersionedProfileReference):
    """Logical versioned reference to domain-owned CodeCraft configuration."""


@dataclass(frozen=True, slots=True)
class RiskProfileRef(_VersionedProfileReference):
    """Logical versioned reference to domain-owned Risk configuration."""


@dataclass(frozen=True, slots=True)
class ScheduleProfileRef(_VersionedProfileReference):
    """Logical versioned reference to domain-owned Schedule configuration."""


@dataclass(frozen=True, slots=True)
class EscalationPolicyRef(_VersionedProfileReference):
    """Logical versioned reference to domain-owned Escalation policy configuration."""


@dataclass(frozen=True, slots=True)
class CollaborationProfileRef(_VersionedProfileReference):
    """Logical versioned reference to domain-owned Collaboration configuration."""


@dataclass(frozen=True, slots=True)
class ObservabilityProfileRef(_VersionedProfileReference):
    """Logical versioned reference to domain-owned Observability configuration."""


def _make_profile_ref_validator(
    label: str,
    ref_type: type[_TProfileRef],
) -> Callable[[object], _TProfileRef]:
    def validate(value: object) -> _TProfileRef:
        if type(value) is not ref_type:
            raise TypeError(
                f"{label} must be {ref_type.__name__}, got {type(value).__name__}"
            )
        return value

    return validate


validate_governance_profile_ref = _make_profile_ref_validator(
    "GovernanceProfileRef",
    GovernanceProfileRef,
)
validate_budget_profile_ref = _make_profile_ref_validator(
    "BudgetProfileRef",
    BudgetProfileRef,
)
validate_memory_profile_ref = _make_profile_ref_validator(
    "MemoryProfileRef",
    MemoryProfileRef,
)
validate_capability_profile_ref = _make_profile_ref_validator(
    "CapabilityProfileRef",
    CapabilityProfileRef,
)
validate_codecraft_profile_ref = _make_profile_ref_validator(
    "CodecraftProfileRef",
    CodecraftProfileRef,
)
validate_risk_profile_ref = _make_profile_ref_validator(
    "RiskProfileRef",
    RiskProfileRef,
)
validate_schedule_profile_ref = _make_profile_ref_validator(
    "ScheduleProfileRef",
    ScheduleProfileRef,
)
validate_escalation_policy_ref = _make_profile_ref_validator(
    "EscalationPolicyRef",
    EscalationPolicyRef,
)
validate_collaboration_profile_ref = _make_profile_ref_validator(
    "CollaborationProfileRef",
    CollaborationProfileRef,
)
validate_observability_profile_ref = _make_profile_ref_validator(
    "ObservabilityProfileRef",
    ObservabilityProfileRef,
)
