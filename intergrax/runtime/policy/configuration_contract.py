# © Artur Czarnecki. All rights reserved.

"""Typed configuration contract bindings and registry (Governed Execution G4B-3)."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Protocol

from pydantic import BaseModel

from intergrax.contracts.policy_catalog import PolicyDefinition, PolicyDefinitionSource
from intergrax.contracts.tool_invocation_control_policy import (
    TOOL_INVOCATION_CONTROL_CONFIGURATION_CONTRACT_ID,
    ToolInvocationControlConfig,
)
from intergrax.runtime.policy.catalog import PolicyCatalog


class ConfigurationContractError(Exception):
    """Base error for configuration contract resolution failures."""


class UnknownConfigurationContractError(ConfigurationContractError):
    """No binding with this contract_id exists."""

    def __init__(self, contract_id: str) -> None:
        self.contract_id = contract_id
        super().__init__(f"unknown configuration contract_id: {contract_id!r}")


class ConfigurationContractConflictError(ConfigurationContractError):
    """Two bindings attempted to own the same contract_id."""

    def __init__(self, contract_id: str) -> None:
        self.contract_id = contract_id
        super().__init__(
            f"duplicate configuration contract_id: {contract_id!r}"
        )


class ConfigurationValidator(Protocol):
    """Typed validation boundary for a single configuration contract."""

    def validate(self, value: object) -> object: ...


def _normalize_contract_id(contract_id: str) -> str:
    normalized = contract_id.strip()
    if not normalized:
        raise ValueError("contract_id must be non-empty")
    return normalized


@dataclass(frozen=True, slots=True)
class ConfigurationContractBinding:
    """Immutable binding from contract_id to typed validation capability (Model B)."""

    contract_id: str
    _validate_fn: Callable[[object], object]

    def __post_init__(self) -> None:
        object.__setattr__(self, "contract_id", _normalize_contract_id(self.contract_id))

    def validate(self, raw_config: object) -> object:
        """Validate raw configuration into an immutable typed configuration object."""
        return self._validate_fn(raw_config)

    @classmethod
    def from_pydantic_model(
        cls,
        contract_id: str,
        model_type: type[BaseModel],
    ) -> ConfigurationContractBinding:
        """Build a binding that validates via a frozen Pydantic model (internal detail)."""

        def _validate(raw_config: object) -> object:
            if isinstance(raw_config, model_type):
                return raw_config
            return model_type.model_validate(raw_config)

        return cls(contract_id=contract_id, _validate_fn=_validate)


def _tool_invocation_control_binding() -> ConfigurationContractBinding:
    return ConfigurationContractBinding.from_pydantic_model(
        TOOL_INVOCATION_CONTROL_CONFIGURATION_CONTRACT_ID,
        ToolInvocationControlConfig,
    )


def built_in_configuration_contract_bindings() -> tuple[ConfigurationContractBinding, ...]:
    """Return canonical built-in configuration contract bindings in deterministic order."""
    return (_tool_invocation_control_binding(),)


def built_in_configuration_contract_ids() -> frozenset[str]:
    """Exact contract_id values reserved by built-in bindings."""
    return frozenset(binding.contract_id for binding in built_in_configuration_contract_bindings())


def build_builtin_configuration_contract_registry() -> ConfigurationContractRegistry:
    """Build the canonical immutable built-in ConfigurationContractRegistry."""
    return build_configuration_contract_registry()


def build_configuration_contract_registry(
    *,
    plugin_bindings: Iterable[ConfigurationContractBinding] = (),
) -> ConfigurationContractRegistry:
    """Compose built-in and validated plugin ConfigurationContractBinding values."""
    return ConfigurationContractRegistry(
        (*built_in_configuration_contract_bindings(), *plugin_bindings),
    )


def validate_builtin_policy_contract_consistency(
    catalog: PolicyCatalog,
    registry: ConfigurationContractRegistry,
) -> None:
    """Prove every built-in PolicyDefinition.configuration_contract_id resolves."""
    for definition in catalog.definitions():
        if definition.source is not PolicyDefinitionSource.BUILT_IN:
            continue
        registry.resolve(definition.configuration_contract_id)


class ConfigurationContractRegistry:
    """Immutable registry mapping exact contract_id to ConfigurationContractBinding."""

    def __init__(
        self,
        bindings: Iterable[ConfigurationContractBinding] = (),
    ) -> None:
        lookup: dict[str, ConfigurationContractBinding] = {}
        ordered: list[ConfigurationContractBinding] = []

        for binding in bindings:
            contract_id = _normalize_contract_id(binding.contract_id)
            if contract_id in lookup:
                raise ConfigurationContractConflictError(contract_id)
            lookup[contract_id] = binding
            ordered.append(binding)

        self._lookup = lookup
        self._bindings = tuple(
            sorted(ordered, key=lambda item: item.contract_id)
        )

    def bindings(self) -> tuple[ConfigurationContractBinding, ...]:
        """Return all bindings in deterministic contract_id order."""
        return self._bindings

    def resolve(self, contract_id: str) -> ConfigurationContractBinding:
        normalized = _normalize_contract_id(contract_id)
        resolved = self._lookup.get(normalized)
        if resolved is None:
            raise UnknownConfigurationContractError(normalized)
        return resolved

    def validate(self, contract_id: str, raw_config: object) -> object:
        """Resolve contract_id and return a typed validated configuration object."""
        return self.resolve(contract_id).validate(raw_config)
