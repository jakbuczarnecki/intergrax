# © Artur Czarnecki. All rights reserved.

"""Model qualification contract for R6 multi-model live qualification.

The execution pipeline depends only on ``ModelQualificationContract`` — not on
concrete providers or registry entries. Extend the matrix by registering new
``RegisteredModelQualification`` instances (or another contract implementation)
without changing the pipeline or cohort executor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from testing_support.decision_e2e.model_matrix.profiles import ModelQualificationProfile


class ModelQualificationContract(Protocol):
    """One model participant in the R6 qualification execution pipeline."""

    @property
    def profile(self) -> ModelQualificationProfile:
        """Immutable profile identity and runtime binding for this model."""

    @property
    def profile_key(self) -> str:
        """Stable registry key used for artifact isolation."""


@dataclass(frozen=True, slots=True)
class RegisteredModelQualification:
    """Registry-backed ``ModelQualificationContract``."""

    profile: ModelQualificationProfile

    @property
    def profile_key(self) -> str:
        return self.profile.profile_key


def contract_for_profile(profile: ModelQualificationProfile) -> RegisteredModelQualification:
    return RegisteredModelQualification(profile=profile)


def contracts_for_profiles(
    profiles: tuple[ModelQualificationProfile, ...],
) -> tuple[RegisteredModelQualification, ...]:
    return tuple(contract_for_profile(profile) for profile in profiles)


def profiles_from_contracts(
    models: tuple[ModelQualificationContract, ...],
) -> tuple[ModelQualificationProfile, ...]:
    return tuple(model.profile for model in models)


__all__ = [
    "ModelQualificationContract",
    "RegisteredModelQualification",
    "contract_for_profile",
    "contracts_for_profiles",
    "profiles_from_contracts",
]
