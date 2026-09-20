# © Artur Czarnecki. All rights reserved.

"""IntegrationProfile runtime materialization — re-exports public contract."""

from __future__ import annotations

from typing import Any, Mapping, Optional

from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.contracts.integration_profile import (
    IntegrationProfile,
    default_lab_profile,
)


def _resolve(
    self: IntegrationProfile,
    category: IntegrationCategory,
    *,
    config: Optional[Mapping[str, Any]] = None,
) -> Any:
    from intergrax.integrations.registry.factory import resolve_from_profile

    return resolve_from_profile(self, category, config=config)


def _lab_stack(cls: type[IntegrationProfile], *, enable_otel: bool = True) -> IntegrationProfile:
    from intergrax.integrations.registry.presets import lab_stack as _lab_stack

    return _lab_stack(enable_otel=enable_otel)


def _legal_stack(cls: type[IntegrationProfile]) -> IntegrationProfile:
    from intergrax.integrations.registry.presets import legal_stack as _legal_stack

    return _legal_stack()


def _research_stack(cls: type[IntegrationProfile]) -> IntegrationProfile:
    from intergrax.integrations.registry.presets import research_stack as _research_stack

    return _research_stack()


def _data_stack(
    cls: type[IntegrationProfile],
    *,
    enable_redis: bool = True,
    enable_qdrant: bool = False,
) -> IntegrationProfile:
    from intergrax.integrations.registry.presets import data_stack as _data_stack

    return _data_stack(enable_redis=enable_redis, enable_qdrant=enable_qdrant)


def _observability_stack(
    cls: type[IntegrationProfile],
    *,
    enable_otel: bool = True,
    enable_grafana_stack: bool = False,
) -> IntegrationProfile:
    from intergrax.integrations.registry.presets import observability_stack as _obs_stack

    return _obs_stack(enable_otel=enable_otel, enable_grafana_stack=enable_grafana_stack)


def _harness_production_stack(
    cls: type[IntegrationProfile],
    *,
    secrets_slug: str = "doppler",
    enable_grafana_stack: bool = True,
) -> IntegrationProfile:
    from intergrax.integrations.registry.presets import harness_production_stack as _prod_stack

    return _prod_stack(
        secrets_slug=secrets_slug,
        enable_grafana_stack=enable_grafana_stack,
    )


setattr(IntegrationProfile, "resolve", _resolve)
setattr(IntegrationProfile, "lab_stack", classmethod(_lab_stack))
setattr(IntegrationProfile, "legal_stack", classmethod(_legal_stack))
setattr(IntegrationProfile, "research_stack", classmethod(_research_stack))
setattr(IntegrationProfile, "data_stack", classmethod(_data_stack))
setattr(IntegrationProfile, "observability_stack", classmethod(_observability_stack))
setattr(IntegrationProfile, "harness_production_stack", classmethod(_harness_production_stack))

__all__ = ["IntegrationProfile", "default_lab_profile"]
