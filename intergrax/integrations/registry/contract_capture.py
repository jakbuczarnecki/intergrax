# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Transitional built-in contract capture — migration-only (P2-003).

**Not canonical authority.** Typed built-in providers must publish explicit
:class:`IntegrationContractSpec` rows via provider-owned declarations and
``register_from_manifest(..., contract_specs=...)``. This module remains only for
built-ins not yet migrated; it must not be extended for new providers.

Reflection-based discovery runs once during ``register_from_manifest`` fallback and
stores immutable metadata on the catalog entry. Contract projection reads stored specs
only — it does not rediscover provider modules.

External plugins must supply explicit :class:`IntegrationContractSpec` rows when they
expose typed platform contracts.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from importlib import import_module
from types import ModuleType
from typing import Any, cast

from intergrax.integrations.core.manifest import IntegrationManifest
from intergrax.integrations.providers.layout import (
    SLUG_CATEGORY,
    categories_for_provider,
    provider_import_path,
)
from intergrax.integrations.registry.contract_spec import (
    IntegrationContractSpec,
    validate_contract_spec_identity,
)
from intergrax.runtime.integrations.categories import OBSERVABILITY_BACKEND_CATEGORY
from intergrax.runtime.integrations.contracts import (
    PlatformIntegrationCapability,
    PlatformIntegrationContract,
    PlatformIntegrationSecurityPosture,
)
from intergrax.runtime.integrations.contract_metadata import contract_for_category
from intergrax.utils import attribute_access

_BUILTIN_SOURCE = "builtin_provider_package"


def capture_builtin_contract_specs(manifest: IntegrationManifest) -> tuple[IntegrationContractSpec, ...]:
    """Derive contract specs for a shipped built-in provider package layout."""
    slug = manifest.slug.strip().lower()
    if slug not in SLUG_CATEGORY:
        return ()
    from intergrax.runtime.integrations.registry_v2 import DEFERRED_LLM_GUARDRAIL_SLUGS

    if slug in DEFERRED_LLM_GUARDRAIL_SLUGS:
        return ()

    specs: list[IntegrationContractSpec] = []
    for category in categories_for_provider(slug):
        specs.append(_capture_builtin_category_spec(slug=slug, category=category))
    return tuple(specs)


def _capture_builtin_category_spec(*, slug: str, category: str) -> IntegrationContractSpec:
    contract_class = contract_for_category(category)
    provider_module_path = provider_import_path(slug, category)
    integration_module = import_module(f"{provider_module_path}.integration")
    bundle_module = import_module(f"{provider_module_path}.bundle")
    integration_class = _find_integration_class(
        integration_module,
        contract_class,
        slug,
    )
    factory_name = _contract_factory_name(slug, category)
    contract_factory = cast(
        Any,
        attribute_access.optional(bundle_module, factory_name),
    )
    if not callable(contract_factory):
        msg = f"{slug}: missing contract factory {factory_name!r} in {bundle_module.__name__}"
        raise ValueError(msg)

    sample = _create_disabled_sample(
        contract_factory,
        slug=slug,
        integration_class=integration_class,
        contract_class=contract_class,
    )
    validate_contract_spec_identity(
        slug=slug,
        spec=IntegrationContractSpec(
            category=category,
            provider_id=sample.provider_id,
            integration_kind=sample.integration_kind,
            contract_class=contract_class,
            integration_class=integration_class,
            contract_factory=contract_factory,
        ),
        observed_provider_id=sample.provider_id,
    )
    capabilities = _capability_values(sample.capabilities)
    runtime_bound = True
    health_supported = (
        PlatformIntegrationCapability.HEALTH_CHECK.value in capabilities and runtime_bound
    )
    metadata: dict[str, object] = {
        "source": _BUILTIN_SOURCE,
        "provider_module": provider_module_path,
        "integration_module": integration_module.__name__,
        "bundle_module": bundle_module.__name__,
        "factory_name": attribute_access.optional_str(
            contract_factory,
            "__name__",
            default=factory_name,
        ),
        "integration_class_name": integration_class.__name__,
        "runtime_binding_supported": runtime_bound,
    }

    return IntegrationContractSpec(
        category=category,
        provider_id=sample.provider_id.strip().lower(),
        integration_kind=sample.integration_kind,
        contract_class=contract_class,
        integration_class=integration_class,
        contract_factory=contract_factory,
        config_class=type(sample.config),
        display_name=sample.display_name or sample.provider_id,
        capabilities=capabilities,
        security_posture=sample.security_posture,
        supports_runtime_binding=runtime_bound,
        supports_health_check=health_supported,
        metadata=metadata,
    )


def _contract_factory_name(slug: str, category: str) -> str:
    if category == OBSERVABILITY_BACKEND_CATEGORY:
        return f"create_{slug}_observability_integration"
    return f"create_{slug}_{category}_integration"


def _find_integration_class(
    module: ModuleType,
    contract_class: type[PlatformIntegrationContract],
    slug: str,
) -> type[PlatformIntegrationContract]:
    candidates = sorted(
        {
            value
            for value in vars(module).values()
            if isinstance(value, type)
            and value.__module__ == module.__name__
            and value.__name__.endswith("Integration")
            and issubclass(value, contract_class)
        },
        key=lambda cls: cls.__name__,
    )
    if len(candidates) != 1:
        names = ", ".join(cls.__name__ for cls in candidates) or "<none>"
        msg = f"{slug}: expected exactly one integration class for {contract_class.__name__}, got {names}"
        raise ValueError(msg)
    return cast(type[PlatformIntegrationContract], candidates[0])


def _create_disabled_sample(
    factory: Any,
    *,
    slug: str,
    integration_class: type[PlatformIntegrationContract],
    contract_class: type[PlatformIntegrationContract],
) -> PlatformIntegrationContract:
    try:
        sample = factory(enabled=False)
    except TypeError as exc:
        msg = f"{slug}: contract factory must accept enabled=False without vendor clients"
        raise ValueError(msg) from exc
    if not isinstance(sample, PlatformIntegrationContract):
        msg = f"{slug}: contract factory must return PlatformIntegrationContract"
        raise ValueError(msg)
    if not isinstance(sample, contract_class):
        msg = f"{slug}: contract factory must return {contract_class.__name__}"
        raise ValueError(msg)
    if not isinstance(sample, integration_class):
        msg = f"{slug}: contract factory must return {integration_class.__name__}"
        raise ValueError(msg)
    if sample.enabled:
        msg = f"{slug}: contract factory enabled=False returned enabled integration"
        raise ValueError(msg)
    if not isinstance(sample.security_posture, PlatformIntegrationSecurityPosture):
        msg = f"{slug}: contract factory must return PlatformIntegrationSecurityPosture"
        raise ValueError(msg)
    return sample


def _capability_values(capabilities: Iterable[Any]) -> tuple[str, ...]:
    values: list[str] = []
    for capability in capabilities:
        value = attribute_access.optional(capability, "value", capability)
        values.append(str(value))
    return tuple(values)


__all__ = [
    "capture_builtin_contract_specs",
]
