# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Registration-time integration contract spec row (catalog contract surface)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING

from intergrax.integrations.contracts.catalog_factory import IntegrationFactory
from intergrax.integrations.contracts.runtime_binding import IntegrationRuntimeBindingSpec

if TYPE_CHECKING:
    from intergrax.runtime.integrations.contracts import (
        PlatformIntegrationConfig,
        PlatformIntegrationContract,
        PlatformIntegrationSecurityPosture,
    )

IntegrationContractFactory = IntegrationFactory


@dataclass(frozen=True, repr=False)
class IntegrationContractSpec:
    """One canonical ``(provider_id, category)`` contract row stored on catalog entries."""

    category: str
    provider_id: str
    integration_kind: str
    contract_class: type[PlatformIntegrationContract]
    integration_class: type[PlatformIntegrationContract]
    security_posture: PlatformIntegrationSecurityPosture
    contract_factory: IntegrationContractFactory = field(compare=False, repr=False)
    config_class: type[PlatformIntegrationConfig] | None = None
    display_name: str = ""
    capabilities: tuple[str, ...] = field(default_factory=tuple)
    supports_runtime_binding: bool = True
    supports_health_check: bool = False
    runtime_binding: IntegrationRuntimeBindingSpec | None = field(
        default=None,
        compare=False,
        repr=False,
    )
    metadata: Mapping[str, object] = field(default_factory=dict, compare=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "capabilities",
            tuple(str(capability) for capability in self.capabilities),
        )
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


__all__ = ["IntegrationContractFactory", "IntegrationContractSpec"]
