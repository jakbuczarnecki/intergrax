# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Provider-neutral Vendor Knowledge source plugin contract and catalog."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType

from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.runtime.vendor_knowledge.live.identity import parse_capability_id

_MAX_PROVIDER_ID_LENGTH = 64
_MAX_SOURCE_KIND_LENGTH = 128
_MAX_REFERENCE_LENGTH = 256
_FORBIDDEN_METADATA_KEYS = frozenset(
    {
        "access_token",
        "connection",
        "connection_ref",
        "credential",
        "credential_ref",
        "credentials",
        "password",
        "secret",
        "tenant",
        "tenant_id",
        "token",
    }
)

type FrozenValue = (
    None
    | bool
    | int
    | float
    | str
    | tuple["FrozenValue", ...]
    | Mapping[str, "FrozenValue"]
)


class VendorKnowledgeMode(StrEnum):
    """Canonical Vendor Knowledge consumption modes."""

    INDEXED = "INDEXED"
    DURABLE = "DURABLE"
    LIVE = "LIVE"


def _non_empty_identifier(value: object, *, field_name: str, max_length: int) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name}_invalid")
    cleaned = value.strip()
    if not cleaned or cleaned != value:
        raise ValueError(f"{field_name}_invalid")
    if len(cleaned) > max_length:
        raise ValueError(f"{field_name}_too_long")
    return cleaned


def _freeze_value(value: object, *, field_name: str) -> FrozenValue:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        frozen: dict[str, FrozenValue] = {}
        for key, item in value.items():
            normalized_key = _non_empty_identifier(
                key,
                field_name=f"{field_name}_key",
                max_length=128,
            )
            if normalized_key.casefold() in _FORBIDDEN_METADATA_KEYS:
                raise ValueError("plugin_metadata_forbidden_field")
            frozen[normalized_key] = _freeze_value(
                item,
                field_name=f"{field_name}.{normalized_key}",
            )
        return MappingProxyType(dict(sorted(frozen.items())))
    if isinstance(value, (list, tuple)):
        return tuple(
            _freeze_value(item, field_name=f"{field_name}_item") for item in value
        )
    if isinstance(value, (set, frozenset)):
        return tuple(
            sorted(
                (_freeze_value(item, field_name=f"{field_name}_item") for item in value),
                key=repr,
            )
        )
    raise ValueError("plugin_metadata_value_invalid")


def _freeze_mapping(
    value: Mapping[str, object],
    *,
    field_name: str,
) -> Mapping[str, FrozenValue]:
    frozen = _freeze_value(value, field_name=field_name)
    if not isinstance(frozen, Mapping):
        raise TypeError(f"{field_name}_invalid")
    return frozen


@dataclass(frozen=True, slots=True)
class VendorKnowledgeSourceIdentity:
    """Stable provider/category/source-kind identity for one source kind."""

    provider_id: str
    integration_category: IntegrationCategory
    source_kind: str

    def __post_init__(self) -> None:
        provider_id = _non_empty_identifier(
            self.provider_id,
            field_name="provider_id",
            max_length=_MAX_PROVIDER_ID_LENGTH,
        )
        source_kind = _non_empty_identifier(
            self.source_kind,
            field_name="source_kind",
            max_length=_MAX_SOURCE_KIND_LENGTH,
        )
        if not isinstance(self.integration_category, IntegrationCategory):
            raise TypeError("integration_category_invalid")
        object.__setattr__(self, "provider_id", provider_id)
        object.__setattr__(self, "source_kind", source_kind)

    @property
    def key(self) -> tuple[str, IntegrationCategory, str]:
        return (self.provider_id, self.integration_category, self.source_kind)


@dataclass(frozen=True, slots=True)
class VendorKnowledgeModeCapability:
    """Declarative, mode-scoped capability and runtime reference."""

    mode: VendorKnowledgeMode
    contract_version: str
    operations: tuple[str, ...]
    runtime_ref: str
    capability_refs: tuple[str, ...] = ()
    constraints: Mapping[str, FrozenValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.mode, VendorKnowledgeMode):
            raise TypeError("mode_invalid")
        contract_version = _non_empty_identifier(
            self.contract_version,
            field_name="mode_contract_version",
            max_length=64,
        )
        runtime_ref = _non_empty_identifier(
            self.runtime_ref,
            field_name="runtime_ref",
            max_length=_MAX_REFERENCE_LENGTH,
        )
        operations = tuple(
            sorted(
                {
                    _non_empty_identifier(
                        operation,
                        field_name="operation",
                        max_length=64,
                    )
                    for operation in self.operations
                }
            )
        )
        if not operations:
            raise ValueError("mode_operations_required")
        capability_refs = tuple(
            sorted(
                {
                    _non_empty_identifier(
                        capability_ref,
                        field_name="capability_ref",
                        max_length=_MAX_REFERENCE_LENGTH,
                    )
                    for capability_ref in self.capability_refs
                }
            )
        )
        if self.mode is VendorKnowledgeMode.LIVE and not capability_refs:
            raise ValueError("live_capability_reference_required")
        constraints = _freeze_mapping(self.constraints, field_name="constraints")
        object.__setattr__(self, "contract_version", contract_version)
        object.__setattr__(self, "runtime_ref", runtime_ref)
        object.__setattr__(self, "operations", operations)
        object.__setattr__(self, "capability_refs", capability_refs)
        object.__setattr__(self, "constraints", constraints)

    def validate_for_source(self, identity: VendorKnowledgeSourceIdentity) -> None:
        """Reject structured live capability IDs belonging to another source."""
        for capability_ref in self.capability_refs:
            if not capability_ref.startswith("vendor."):
                continue
            try:
                provider_id, source_kind, _operation = parse_capability_id(capability_ref)
            except (TypeError, ValueError) as exc:
                raise ValueError("capability_reference_invalid") from exc
            if provider_id != identity.provider_id or source_kind != identity.source_kind:
                raise ValueError("capability_reference_source_mismatch")


@dataclass(frozen=True, slots=True)
class VendorKnowledgeSourcePlugin:
    """Immutable declarative descriptor composing the three mode runtimes."""

    identity: VendorKnowledgeSourceIdentity
    capabilities: tuple[VendorKnowledgeModeCapability, ...]
    contract_version: str = "vendor-knowledge.source-plugin.v1"
    metadata: Mapping[str, FrozenValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.identity, VendorKnowledgeSourceIdentity):
            raise TypeError("source_identity_required")
        contract_version = _non_empty_identifier(
            self.contract_version,
            field_name="plugin_contract_version",
            max_length=64,
        )
        capabilities = tuple(self.capabilities)
        if not capabilities:
            raise ValueError("source_mode_capability_required")
        if any(
            not isinstance(capability, VendorKnowledgeModeCapability)
            for capability in capabilities
        ):
            raise ValueError("source_mode_capability_invalid")
        modes = [capability.mode for capability in capabilities]
        if len(set(modes)) != len(modes):
            raise ValueError("duplicate_source_mode")
        normalized_capabilities = tuple(
            sorted(capabilities, key=lambda capability: capability.mode.value)
        )
        capability_refs: set[str] = set()
        for capability in normalized_capabilities:
            capability.validate_for_source(self.identity)
            duplicate_refs = capability_refs.intersection(capability.capability_refs)
            if duplicate_refs:
                raise ValueError("duplicate_capability_identity")
            capability_refs.update(capability.capability_refs)
        metadata = _freeze_mapping(self.metadata, field_name="metadata")
        object.__setattr__(self, "contract_version", contract_version)
        object.__setattr__(self, "capabilities", normalized_capabilities)
        object.__setattr__(self, "metadata", metadata)

    def supports(self, mode: VendorKnowledgeMode) -> bool:
        return self.capability(mode) is not None

    def capability(
        self,
        mode: VendorKnowledgeMode,
    ) -> VendorKnowledgeModeCapability | None:
        if not isinstance(mode, VendorKnowledgeMode):
            raise TypeError("mode_invalid")
        return next(
            (capability for capability in self.capabilities if capability.mode is mode),
            None,
        )

    def capabilities_for(
        self,
        mode: VendorKnowledgeMode,
    ) -> tuple[VendorKnowledgeModeCapability, ...]:
        capability = self.capability(mode)
        return () if capability is None else (capability,)


class VendorKnowledgeSourcePluginNotFound(LookupError):
    """Raised when a source identity is not present in the catalog."""

    def __init__(self, identity: VendorKnowledgeSourceIdentity) -> None:
        super().__init__(f"Vendor Knowledge source plugin is not registered: {identity.key!r}")
        self.identity = identity


class VendorKnowledgeSourcePluginConflict(ValueError):
    """Raised when a source identity is registered with different content."""


class VendorKnowledgeSourcePluginRegistry:
    """Authoritative provider-neutral source plugin discovery catalog."""

    def __init__(self) -> None:
        self._plugins: dict[
            tuple[str, IntegrationCategory, str],
            VendorKnowledgeSourcePlugin,
        ] = {}

    def register(self, plugin: VendorKnowledgeSourcePlugin) -> None:
        if not isinstance(plugin, VendorKnowledgeSourcePlugin):
            raise TypeError("source_plugin_invalid")
        key = plugin.identity.key
        existing = self._plugins.get(key)
        if existing is None:
            self._plugins[key] = plugin
            return
        if existing == plugin:
            return
        raise VendorKnowledgeSourcePluginConflict(
            f"conflicting Vendor Knowledge source plugin registration: {key!r}"
        )

    def get(
        self,
        *,
        provider_id: str,
        integration_category: IntegrationCategory,
        source_kind: str,
    ) -> VendorKnowledgeSourcePlugin | None:
        identity = VendorKnowledgeSourceIdentity(
            provider_id=provider_id,
            integration_category=integration_category,
            source_kind=source_kind,
        )
        return self._plugins.get(identity.key)

    def require(self, identity: VendorKnowledgeSourceIdentity) -> VendorKnowledgeSourcePlugin:
        if not isinstance(identity, VendorKnowledgeSourceIdentity):
            raise TypeError("source_identity_required")
        plugin = self._plugins.get(identity.key)
        if plugin is None:
            raise VendorKnowledgeSourcePluginNotFound(identity)
        return plugin

    def lookup(
        self,
        identity: VendorKnowledgeSourceIdentity,
    ) -> VendorKnowledgeSourcePlugin | None:
        if not isinstance(identity, VendorKnowledgeSourceIdentity):
            raise TypeError("source_identity_required")
        return self._plugins.get(identity.key)

    def list_plugins(self) -> tuple[VendorKnowledgeSourcePlugin, ...]:
        return tuple(
            self._plugins[key]
            for key in sorted(
                self._plugins,
                key=lambda item: (item[0], item[1].value, item[2]),
            )
        )

    def list_source_plugins(self) -> tuple[VendorKnowledgeSourcePlugin, ...]:
        return self.list_plugins()

    def list_source_kinds(self) -> tuple[VendorKnowledgeSourceIdentity, ...]:
        return tuple(plugin.identity for plugin in self.list_plugins())

    def supports(
        self,
        identity: VendorKnowledgeSourceIdentity,
        mode: VendorKnowledgeMode,
    ) -> bool:
        return self.require(identity).supports(mode)

    def list_capabilities(
        self,
        identity: VendorKnowledgeSourceIdentity,
        mode: VendorKnowledgeMode,
    ) -> tuple[VendorKnowledgeModeCapability, ...]:
        return self.require(identity).capabilities_for(mode)


__all__ = [
    "VendorKnowledgeMode",
    "VendorKnowledgeModeCapability",
    "VendorKnowledgeSourceIdentity",
    "VendorKnowledgeSourcePlugin",
    "VendorKnowledgeSourcePluginConflict",
    "VendorKnowledgeSourcePluginNotFound",
    "VendorKnowledgeSourcePluginRegistry",
]
