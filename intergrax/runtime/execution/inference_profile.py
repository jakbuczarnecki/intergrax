# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Execution-owned inference profile identity and adapter resolution (DS-DELIB-02)."""

from __future__ import annotations

from types import MappingProxyType
from typing import NewType, Protocol

from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter

InferenceProfileId = NewType("InferenceProfileId", str)


class InferenceProfileError(RuntimeError):
    """Base failure for inference profile resolution."""


class InferenceProfileNotFoundError(InferenceProfileError):
    """Raised when an explicit profile id is not registered in host composition."""

    def __init__(self, profile_id: InferenceProfileId) -> None:
        super().__init__(f"inference profile not found: {profile_id!r}")
        self.profile_id = profile_id


class InferenceProfileAlreadyRegisteredError(InferenceProfileError):
    """Raised when host composition registers the same profile id twice."""

    def __init__(self, profile_id: InferenceProfileId) -> None:
        super().__init__(f"inference profile already registered: {profile_id!r}")
        self.profile_id = profile_id


class InferenceProfileResolutionError(InferenceProfileError):
    """Raised when explicit profile selection cannot be resolved."""


def validate_inference_profile_id(value: object) -> InferenceProfileId:
    """Validate a logical inference profile identity for Execution requests."""
    if type(value) is not str:
        raise TypeError(
            f"InferenceProfileId must be str, got {type(value).__name__}",
        )
    if not value or not value.strip():
        raise ValueError(
            "InferenceProfileId must be non-empty and not whitespace-only",
        )
    if value != value.strip():
        raise ValueError(
            "InferenceProfileId must not contain leading or trailing whitespace",
        )
    return InferenceProfileId(value)


class InferenceProfileResolver(Protocol):
    """Resolve a logical inference profile to a host-owned adapter."""

    def resolve(self, profile_id: InferenceProfileId) -> LLMAdapter:
        ...


class InferenceProfileCatalog:
    """Immutable host composition mapping logical profiles to adapters."""

    __slots__ = ("_adapters",)

    def __init__(
        self,
        profiles: tuple[tuple[str | InferenceProfileId, LLMAdapter], ...],
    ) -> None:
        validated: dict[InferenceProfileId, LLMAdapter] = {}
        for raw_profile_id, adapter in profiles:
            profile_id = validate_inference_profile_id(raw_profile_id)
            if profile_id in validated:
                raise InferenceProfileAlreadyRegisteredError(profile_id)
            validated[profile_id] = adapter
        self._adapters = MappingProxyType(validated)

    def resolve(self, profile_id: InferenceProfileId) -> LLMAdapter:
        try:
            return self._adapters[profile_id]
        except KeyError as exc:
            raise InferenceProfileNotFoundError(profile_id) from exc
