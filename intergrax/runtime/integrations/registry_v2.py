# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Contract registry projection derived from the canonical integration catalog.

``intergrax.integrations.registry.catalog`` is the single authoritative provider
registration lifecycle. This module exposes an immutable read model for typed
contract/capability inspection and compatibility validation.

It does **not** replace catalog factories, bootstrap registration, or
``IntegrationProfile`` runtime resolution.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TypeAlias, cast

from intergrax.integrations.contracts.base import UnknownIntegrationError
from intergrax.integrations.providers.layout import SLUG_CATEGORY
from intergrax.integrations.registry.catalog import catalog_snapshot, get_entry
from intergrax.integrations.registry.contract_spec import IntegrationContractSpec
from intergrax.runtime.integrations.categories import PROVIDER_CATEGORY_CONTRACT_REGISTRY
from intergrax.runtime.integrations.contract_metadata import (
    IntegrationContractMetadataError,
    contract_for_category,
    expected_integration_kind_for_category,
    normalize_contract_identity as _normalize_contract_identity,
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


class IntegrationRegistryError(ValueError):
    """Base error for contract projection validation failures."""


class IntegrationContractProjectionError(IntegrationRegistryError):
    """Raised when canonical catalog metadata cannot be projected safely."""


class DuplicateIntegrationRegistrationError(IntegrationRegistryError):
    """Raised when provider_id + category is already registered in a projection."""


class MissingIntegrationRegistrationError(IntegrationRegistryError):
    """Raised when provider_id + category cannot be found in a projection."""


def _normalize_identity(value: str, field_name: str) -> str:
    try:
        return _normalize_contract_identity(value, field_name)
    except IntegrationContractMetadataError as exc:
        raise IntegrationRegistryError(str(exc)) from exc


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
            msg = f"Unknown integration category for contract projection: {category!r}"
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
            msg = f"{slug}: contract projection registrations must be disabled by default"
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
    """Immutable in-memory projection of canonical contract registrations."""

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
        key = (
            _normalize_identity(provider_id, "provider_id"),
            _normalize_identity(category, "category"),
        )
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


def registration_from_contract_spec(*, slug: str, spec: IntegrationContractSpec) -> IntegrationRegistration:
    """Project one canonical contract spec into an immutable registration row."""
    normalized_slug = _normalize_identity(slug, "slug")
    normalized_provider = _normalize_identity(spec.provider_id, "provider_id")
    if normalized_provider != normalized_slug and spec.metadata.get("allow_provider_slug_alias") is not True:
        msg = (
            f"Integration contract identity mismatch for slug {normalized_slug!r}: "
            f"registered provider_id={normalized_provider!r}"
        )
        raise IntegrationContractProjectionError(msg)

    return IntegrationRegistration(
        provider_id=spec.provider_id,
        slug=normalized_slug,
        category=_normalize_identity(spec.category, "category"),
        integration_kind=spec.integration_kind,
        contract_class=spec.contract_class,
        integration_class=spec.integration_class,
        factory=cast(IntegrationFactory, spec.contract_factory),
        config_class=spec.config_class,
        display_name=spec.display_name or spec.provider_id,
        capabilities=spec.capabilities,
        security_posture=spec.security_posture or PlatformIntegrationSecurityPosture(),
        default_enabled=False,
        supports_health_check=spec.supports_health_check,
        supports_runtime_binding=spec.supports_runtime_binding,
        metadata=spec.metadata,
    )


def build_integration_registration(
    slug: str,
    *,
    category: str | None = None,
    factory: IntegrationFactory | None = None,
    integration_class: type[PlatformIntegrationContract] | None = None,
    metadata: Mapping[str, object] | None = None,
    supports_runtime_binding: bool | None = None,
    supports_health_check: bool | None = None,
) -> IntegrationRegistration:
    """Build a contract projection row from canonical catalog metadata."""
    normalized_slug = _normalize_identity(slug, "slug")
    try:
        entry = get_entry(normalized_slug)
    except UnknownIntegrationError as exc:
        msg = f"Integration slug {normalized_slug!r} is not registered in the canonical catalog"
        raise MissingIntegrationRegistrationError(msg) from exc

    if not entry.contract_specs:
        msg = (
            f"Integration slug {normalized_slug!r} has no contract projection metadata; "
            "register typed contract specs during canonical registration"
        )
        raise MissingIntegrationRegistrationError(msg)

    if category is None:
        if len(entry.contract_specs) == 1:
            spec = entry.contract_specs[0]
        else:
            primary_category = entry.categories[0].value
            matches = tuple(
                candidate
                for candidate in entry.contract_specs
                if candidate.category == primary_category
            )
            if len(matches) != 1:
                msg = (
                    f"Integration slug {normalized_slug!r} exposes multiple categories; "
                    "pass category= explicitly"
                )
                raise IntegrationRegistryError(msg)
            spec = matches[0]
    else:
        normalized_category = _normalize_identity(category, "category")
        matches = tuple(spec for spec in entry.contract_specs if spec.category == normalized_category)
        if len(matches) != 1:
            msg = (
                f"No canonical contract spec for slug {normalized_slug!r} / category {normalized_category!r}"
            )
            raise MissingIntegrationRegistrationError(msg)
        spec = matches[0]

    registration = registration_from_contract_spec(slug=normalized_slug, spec=spec)

    if factory is not None or integration_class is not None or metadata or supports_runtime_binding is not None or supports_health_check is not None:
        # Test/local override path — still derived from canonical spec by default.
        return IntegrationRegistration(
            provider_id=registration.provider_id,
            slug=registration.slug,
            category=registration.category,
            integration_kind=registration.integration_kind,
            contract_class=registration.contract_class,
            integration_class=integration_class or registration.integration_class,
            factory=factory or registration.factory,
            config_class=registration.config_class,
            display_name=registration.display_name,
            capabilities=registration.capabilities,
            security_posture=registration.security_posture,
            default_enabled=False,
            supports_health_check=(
                supports_health_check
                if supports_health_check is not None
                else registration.supports_health_check
            ),
            supports_runtime_binding=(
                supports_runtime_binding
                if supports_runtime_binding is not None
                else registration.supports_runtime_binding
            ),
            metadata={**dict(registration.metadata), **dict(metadata or {})},
        )
    return registration


def build_contract_registry_snapshot(
    slugs: Iterable[str] | None = None,
) -> IntegrationRegistry:
    """Build an immutable contract projection snapshot from the canonical catalog."""
    snapshot = catalog_snapshot()
    if slugs is not None:
        selected_slugs = {
            _normalize_identity(slug, "slug")
            for slug in slugs
        }
        snapshot = {slug: entry for slug, entry in snapshot.items() if slug in selected_slugs}

    registry = IntegrationRegistry()
    for slug in sorted(snapshot):
        entry = snapshot[slug]
        for spec in entry.contract_specs:
            registry.register(registration_from_contract_spec(slug=slug, spec=spec))
    return registry


def build_contract_registry(
    slugs: Iterable[str] | None = None,
) -> IntegrationRegistry:
    """Backward-compatible alias for :func:`build_contract_registry_snapshot`."""
    return build_contract_registry_snapshot(slugs=slugs)


def non_deferred_provider_slugs() -> tuple[str, ...]:
    """Provider slugs with canonical contract projection coverage."""
    return tuple(
        sorted(
            slug
            for slug, entry in catalog_snapshot().items()
            if entry.contract_specs
        )
    )


__all__ = [
    "DuplicateIntegrationRegistrationError",
    "IntegrationCapability",
    "IntegrationContractProjectionError",
    "IntegrationFactory",
    "IntegrationRegistration",
    "IntegrationRegistry",
    "IntegrationRegistryError",
    "IntegrationSecurityPosture",
    "MissingIntegrationRegistrationError",
    "RegistrationKey",
    "build_contract_registry",
    "build_contract_registry_snapshot",
    "build_integration_registration",
    "contract_for_category",
    "expected_integration_kind_for_category",
    "get_integration_registration",
    "list_by_category",
    "list_by_provider",
    "list_integration_registrations",
    "non_deferred_provider_slugs",
    "register_integration",
    "registration_from_contract_spec",
]
