# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Canonical registration-time contract metadata for integration providers."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any


IntegrationContractFactory = Callable[..., Any]


@dataclass(frozen=True, repr=False)
class IntegrationContractSpec:
    """One canonical ``(provider_id, category)`` contract row stored on catalog entries."""

    category: str
    provider_id: str
    integration_kind: str
    contract_class: type[Any]
    integration_class: type[Any]
    contract_factory: IntegrationContractFactory = field(compare=False, repr=False)
    config_class: type[Any] | None = None
    display_name: str = ""
    capabilities: tuple[str, ...] = field(default_factory=tuple)
    security_posture: Any = None
    supports_runtime_binding: bool = True
    supports_health_check: bool = False
    metadata: Mapping[str, object] = field(default_factory=dict, compare=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "capabilities", tuple(str(capability) for capability in self.capabilities))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


def validate_contract_spec_identity(
    *,
    slug: str,
    spec: IntegrationContractSpec,
    observed_provider_id: str,
) -> None:
    """Fail when typed integration identity drifts from canonical registration metadata."""
    normalized_slug = slug.strip().lower()
    normalized_provider = observed_provider_id.strip().lower()
    normalized_spec = spec.provider_id.strip().lower()
    if normalized_spec != normalized_provider:
        msg = (
            f"Integration contract identity mismatch for slug {normalized_slug!r}: "
            f"registered provider_id={normalized_spec!r}, integration reports {normalized_provider!r}"
        )
        raise ValueError(msg)


__all__ = [
    "IntegrationContractFactory",
    "IntegrationContractSpec",
    "validate_contract_spec_identity",
]
