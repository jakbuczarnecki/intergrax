# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Contract-aware integration registry v2 (INTEGRATIONS-3A).

This module is intentionally additive. It models provider/category registrations
without replacing the existing catalog, bootstrapping providers, resolving runtime
bindings, loading secrets, or constructing vendor clients.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from importlib import import_module
from types import MappingProxyType, ModuleType
from typing import Any, TypeAlias, cast

from intergrax.integrations.providers.layout import SLUG_CATEGORY, provider_import_path
from intergrax.runtime.integrations.categories import (
    OBSERVABILITY_BACKEND_CATEGORY,
    OBSERVABILITY_VENDOR_INTEGRATION_KIND,
    PROVIDER_CATEGORY_CONTRACT_REGISTRY,
)
from intergrax.runtime.integrations.contracts import (
    PlatformIntegrationCapability,
    PlatformIntegrationConfig,
    PlatformIntegrationContract,
    PlatformIntegrationSecurityPosture,
)

IntegrationCapability: TypeAlias = PlatformIntegrationCapability
IntegrationSecurityPosture: TypeAlias = PlatformIntegrationSecurityPosture
IntegrationFactory: TypeAlias = Callable[..., PlatformIntegrationContract]
RegistrationKey: TypeAlias = tuple[str, str]

DEFERRED_LLM_GUARDRAIL_SLUGS: frozenset[str] = frozenset(
    {
        "llm_guard",
        "guardrails_ai",
        "nemo_guardrails",
        "openguardrails",
        "presidio",
        "llama_guard",
        "lakera",
        "azure_content_safety",
        "bedrock_guardrails",
    }
)


class IntegrationRegistryError(ValueError):
    """Base error for contract registry v2 validation failures."""


class DuplicateIntegrationRegistrationError(IntegrationRegistryError):
    """Raised when provider_id + category is already registered."""


class MissingIntegrationRegistrationError(IntegrationRegistryError):
    """Raised when provider_id + category cannot be found."""


@dataclass(frozen=True, repr=False)
class IntegrationRegistration:
    """Immutable, inspectable provider/category registration metadata."""

    provider_id: str
    slug: str
    category: str
    integration_kind: str
    contract_class: type[PlatformIntegrationContract]
    integration_class: type[PlatformIntegrationContract]
    factory: IntegrationFactory = field(compare=False, repr=False)
    config_class: type[PlatformIntegrationConfig] | None = None
    display_name: str = ""
    capabilities: tuple[str, ...] = field(default_factory=tuple)
    security_posture: IntegrationSecurityPosture = field(
        default_factory=IntegrationSecurityPosture,
        compare=False,
        repr=False,
    )
    default_enabled: bool = False
    supports_health_check: bool = False
    supports_runtime_binding: bool = True
    metadata: Mapping[str, object] = field(default_factory=dict, compare=False, repr=False)

    def __post_init__(self) -> None:
        provider_id = _normalize_identity(self.provider_id, "provider_id")
        slug = _normalize_identity(self.slug, "slug")
        category = _normalize_identity(self.category, "category")
        integration_kind = _normalize_identity(self.integration_kind, "integration_kind")

        expected_contract = PROVIDER_CATEGORY_CONTRACT_REGISTRY.get(category)
        if expected_contract is None:
            msg = f"Unknown integration category for registry v2: {category!r}"
            raise IntegrationRegistryError(msg)
        if self.contract_class is not expected_contract:
            msg = (
                f"{slug}: contract_class must be {expected_contract.__name__} "
                f"for category {category!r}, got {self.contract_class.__name__}"
            )
            raise IntegrationRegistryError(msg)
        if not issubclass(self.contract_class, PlatformIntegrationContract):
            msg = f"{slug}: contract_class must derive from PlatformIntegrationContract"
            raise IntegrationRegistryError(msg)
        if not issubclass(self.integration_class, self.contract_class):
            msg = (
                f"{slug}: integration_class {self.integration_class.__name__} must derive from "
                f"{self.contract_class.__name__}"
            )
            raise IntegrationRegistryError(msg)
        expected_kind = expected_integration_kind_for_category(category)
        if integration_kind != expected_kind:
            msg = (
                f"{slug}: integration_kind must be {expected_kind!r} for category {category!r}, "
                f"got {integration_kind!r}"
            )
            raise IntegrationRegistryError(msg)
        if not callable(self.factory):
            msg = f"{slug}: factory must be callable"
            raise IntegrationRegistryError(msg)
        if self.config_class is not None and not issubclass(self.config_class, PlatformIntegrationConfig):
            msg = f"{slug}: config_class must derive from PlatformIntegrationConfig"
            raise IntegrationRegistryError(msg)
        if self.default_enabled:
            msg = f"{slug}: registry v2 registrations must be disabled by default"
            raise IntegrationRegistryError(msg)
        if not isinstance(self.security_posture, PlatformIntegrationSecurityPosture):
            msg = f"{slug}: security_posture must be PlatformIntegrationSecurityPosture"
            raise IntegrationRegistryError(msg)

        object.__setattr__(self, "provider_id", provider_id)
        object.__setattr__(self, "slug", slug)
        object.__setattr__(self, "category", category)
        object.__setattr__(self, "integration_kind", integration_kind)
        object.__setattr__(self, "capabilities", tuple(str(capability) for capability in self.capabilities))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def key(self) -> RegistrationKey:
        """Stable registry identity: one provider_id per category."""
        return (self.provider_id, self.category)

    def __repr__(self) -> str:
        return (
            "IntegrationRegistration("
            f"provider_id={self.provider_id!r}, "
            f"slug={self.slug!r}, "
            f"category={self.category!r}, "
            f"integration_kind={self.integration_kind!r}, "
            f"contract_class={self.contract_class.__name__}, "
            f"integration_class={self.integration_class.__name__}, "
            f"default_enabled={self.default_enabled!r}, "
            f"supports_runtime_binding={self.supports_runtime_binding!r})"
        )


class IntegrationRegistry:
    """Deterministic in-memory registry for contract-aware provider registrations."""

    def __init__(self, registrations: Iterable[IntegrationRegistration] = ()) -> None:
        self._registrations: dict[RegistrationKey, IntegrationRegistration] = {}
        for registration in registrations:
            self.register(registration)

    def register(self, registration: IntegrationRegistration) -> IntegrationRegistration:
        key = registration.key
        if key in self._registrations:
            provider_id, category = key
            msg = f"Integration registration already exists for {provider_id!r} / {category!r}"
            raise DuplicateIntegrationRegistrationError(msg)
        self._registrations[key] = registration
        return registration

    def get(self, *, provider_id: str, category: str) -> IntegrationRegistration:
        key = (_normalize_identity(provider_id, "provider_id"), _normalize_identity(category, "category"))
        try:
            return self._registrations[key]
        except KeyError as exc:
            msg = f"No integration registration for {key[0]!r} / {key[1]!r}"
            raise MissingIntegrationRegistrationError(msg) from exc

    def list_all(self) -> tuple[IntegrationRegistration, ...]:
        return tuple(self._registrations[key] for key in sorted(self._registrations))

    def list_by_category(self, category: str) -> tuple[IntegrationRegistration, ...]:
        normalized = _normalize_identity(category, "category")
        return tuple(
            registration
            for registration in self.list_all()
            if registration.category == normalized
        )

    def list_by_provider(self, provider_id: str) -> tuple[IntegrationRegistration, ...]:
        normalized = _normalize_identity(provider_id, "provider_id")
        return tuple(
            registration
            for registration in self.list_all()
            if registration.provider_id == normalized
        )

    def __len__(self) -> int:
        return len(self._registrations)

    def __contains__(self, key: object) -> bool:
        return key in self._registrations


# Backward-compatible functional names for callers/tests that prefer module helpers.
def register_integration(registry: IntegrationRegistry, registration: IntegrationRegistration) -> IntegrationRegistration:
    return registry.register(registration)


def get_integration_registration(
    registry: IntegrationRegistry,
    *,
    provider_id: str,
    category: str,
) -> IntegrationRegistration:
    return registry.get(provider_id=provider_id, category=category)


def list_integration_registrations(registry: IntegrationRegistry) -> tuple[IntegrationRegistration, ...]:
    return registry.list_all()


def list_by_category(registry: IntegrationRegistry, category: str) -> tuple[IntegrationRegistration, ...]:
    return registry.list_by_category(category)


def list_by_provider(registry: IntegrationRegistry, provider_id: str) -> tuple[IntegrationRegistration, ...]:
    return registry.list_by_provider(provider_id)


def build_integration_registration(
    slug: str,
    *,
    factory: IntegrationFactory | None = None,
    integration_class: type[PlatformIntegrationContract] | None = None,
    metadata: Mapping[str, object] | None = None,
) -> IntegrationRegistration:
    """Build a registry v2 registration from a provider package without enabling it."""
    normalized_slug = _normalize_identity(slug, "slug")
    try:
        category = SLUG_CATEGORY[normalized_slug]
    except KeyError as exc:
        msg = f"Unknown provider slug for registry v2: {normalized_slug!r}"
        raise IntegrationRegistryError(msg) from exc

    contract_class = contract_for_category(category)
    provider_module_path = provider_import_path(normalized_slug)
    integration_module = import_module(f"{provider_module_path}.integration")
    bundle_module = import_module(f"{provider_module_path}.bundle")
    resolved_integration_class = integration_class or _find_integration_class(
        integration_module,
        contract_class,
        normalized_slug,
    )
    factory_name = _contract_factory_name(normalized_slug, category)
    resolved_factory = factory or cast(IntegrationFactory, getattr(bundle_module, factory_name))

    sample = _create_disabled_sample(
        resolved_factory,
        slug=normalized_slug,
        integration_class=resolved_integration_class,
        contract_class=contract_class,
    )
    capabilities = _capability_values(sample.capabilities)
    safe_metadata: dict[str, object] = {
        "source": "provider_package",
        "provider_module": provider_module_path,
        "integration_module": integration_module.__name__,
        "bundle_module": bundle_module.__name__,
        "factory_name": getattr(resolved_factory, "__name__", factory_name),
        "integration_class_name": resolved_integration_class.__name__,
    }
    if metadata:
        safe_metadata.update(metadata)

    return IntegrationRegistration(
        provider_id=sample.provider_id,
        slug=normalized_slug,
        category=category,
        integration_kind=sample.integration_kind,
        contract_class=contract_class,
        integration_class=resolved_integration_class,
        factory=resolved_factory,
        config_class=type(sample.config),
        display_name=sample.display_name or sample.provider_id,
        capabilities=capabilities,
        security_posture=sample.security_posture,
        default_enabled=sample.enabled,
        supports_health_check=PlatformIntegrationCapability.HEALTH_CHECK.value in capabilities,
        supports_runtime_binding=True,
        metadata=safe_metadata,
    )


def build_contract_registry(
    slugs: Iterable[str] | None = None,
    *,
    exclude_deferred: bool = True,
) -> IntegrationRegistry:
    """Build a contract-aware registry snapshot without modifying global bootstrap state."""
    selected_slugs = tuple(slugs) if slugs is not None else tuple(sorted(SLUG_CATEGORY))
    registry = IntegrationRegistry()
    for slug in selected_slugs:
        normalized_slug = _normalize_identity(slug, "slug")
        if exclude_deferred and normalized_slug in DEFERRED_LLM_GUARDRAIL_SLUGS:
            continue
        registry.register(build_integration_registration(normalized_slug))
    return registry


def non_deferred_provider_slugs() -> tuple[str, ...]:
    """Provider slugs covered by registry v2 compatibility checks."""
    return tuple(sorted(slug for slug in SLUG_CATEGORY if slug not in DEFERRED_LLM_GUARDRAIL_SLUGS))


def expected_integration_kind_for_category(category: str) -> str:
    normalized = _normalize_identity(category, "category")
    if normalized == OBSERVABILITY_BACKEND_CATEGORY:
        return OBSERVABILITY_VENDOR_INTEGRATION_KIND
    return normalized


def contract_for_category(category: str) -> type[PlatformIntegrationContract]:
    normalized = _normalize_identity(category, "category")
    try:
        return PROVIDER_CATEGORY_CONTRACT_REGISTRY[normalized]
    except KeyError as exc:
        msg = f"Unknown integration category for registry v2: {normalized!r}"
        raise IntegrationRegistryError(msg) from exc


def _normalize_identity(value: str, field_name: str) -> str:
    normalized = value.strip().lower()
    if not normalized:
        msg = f"{field_name} must be a non-empty string"
        raise IntegrationRegistryError(msg)
    return normalized


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
        (
            value
            for value in vars(module).values()
            if isinstance(value, type)
            and value.__module__ == module.__name__
            and value.__name__.endswith("Integration")
            and issubclass(value, contract_class)
        ),
        key=lambda cls: cls.__name__,
    )
    if len(candidates) != 1:
        names = ", ".join(cls.__name__ for cls in candidates) or "<none>"
        msg = f"{slug}: expected exactly one integration class for {contract_class.__name__}, got {names}"
        raise IntegrationRegistryError(msg)
    return cast(type[PlatformIntegrationContract], candidates[0])


def _create_disabled_sample(
    factory: IntegrationFactory,
    *,
    slug: str,
    integration_class: type[PlatformIntegrationContract],
    contract_class: type[PlatformIntegrationContract],
) -> PlatformIntegrationContract:
    try:
        sample = factory(enabled=False)
    except TypeError as exc:
        msg = f"{slug}: contract factory must accept enabled=False without vendor clients"
        raise IntegrationRegistryError(msg) from exc
    if not isinstance(sample, PlatformIntegrationContract):
        msg = f"{slug}: contract factory must return PlatformIntegrationContract"
        raise IntegrationRegistryError(msg)
    if not isinstance(sample, contract_class):
        msg = f"{slug}: contract factory must return {contract_class.__name__}"
        raise IntegrationRegistryError(msg)
    if not isinstance(sample, integration_class):
        msg = f"{slug}: contract factory must return {integration_class.__name__}"
        raise IntegrationRegistryError(msg)
    if sample.enabled:
        msg = f"{slug}: contract factory enabled=False returned enabled integration"
        raise IntegrationRegistryError(msg)
    return sample


def _capability_values(capabilities: Iterable[Any]) -> tuple[str, ...]:
    values: list[str] = []
    for capability in capabilities:
        value = getattr(capability, "value", capability)
        values.append(str(value))
    return tuple(values)


__all__ = [
    "DEFERRED_LLM_GUARDRAIL_SLUGS",
    "DuplicateIntegrationRegistrationError",
    "IntegrationCapability",
    "IntegrationFactory",
    "IntegrationRegistration",
    "IntegrationRegistry",
    "IntegrationRegistryError",
    "IntegrationSecurityPosture",
    "MissingIntegrationRegistrationError",
    "RegistrationKey",
    "build_contract_registry",
    "build_integration_registration",
    "contract_for_category",
    "expected_integration_kind_for_category",
    "get_integration_registration",
    "list_by_category",
    "list_by_provider",
    "list_integration_registrations",
    "non_deferred_provider_slugs",
    "register_integration",
]
