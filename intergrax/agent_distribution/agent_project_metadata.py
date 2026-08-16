# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Neutral agent project metadata provider port (AP-7) and package pyproject parse."""

from __future__ import annotations

import tomllib
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator

from intergrax.agent_distribution.agent_capability_metadata import AgentCapabilityDescriptor

_NON_EMPTY = Field(min_length=1)
_AGENT_TOOL_TABLE = ("tool", "intergrax", "agent")


def _strip_required(value: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError("must be non-empty")
    return normalized


class AgentProjectMetadataParseError(ValueError):
    """Raised when agent package pyproject metadata cannot be parsed."""


class AgentPackageContractDeclaration(BaseModel):
    """Declarative non-executable contract row owned by a Tier-2 package."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    contract_id: str = _NON_EMPTY
    capabilities: tuple[str, ...] = ()
    skill_ids: tuple[str, ...] = ()
    tool_ids: tuple[str, ...] = ()

    @field_validator("contract_id")
    @classmethod
    def _strip_contract_id(cls, value: str) -> str:
        return _strip_required(value)

    @field_validator("capabilities", "skill_ids", "tool_ids")
    @classmethod
    def _strip_tuple_items(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(_strip_required(item) for item in value)


class AgentProjectMetadata(BaseModel):
    """Parsed installed-agent project metadata — no filesystem access."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    distribution_package_id: str = _NON_EMPTY
    dependencies: tuple[str, ...] = ()
    package_version: str | None = None
    declared_contracts: tuple[AgentPackageContractDeclaration, ...] = ()

    @field_validator("distribution_package_id")
    @classmethod
    def _strip_distribution_package_id(cls, value: str) -> str:
        return _strip_required(value)

    @field_validator("package_version")
    @classmethod
    def _strip_optional_package_version(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _strip_required(value)


class AgentProjectMetadataProvider(Protocol):
    """Resolve authoritative installed-agent metadata by opaque ref."""

    def get_metadata(self, metadata_ref: str) -> AgentProjectMetadata | None:
        """Load parsed metadata for one installed agent artifact."""


def _require_mapping(value: object, *, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise AgentProjectMetadataParseError(f"{field_name} must be a mapping")
    return value


def _optional_str(value: object, *, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise AgentProjectMetadataParseError(f"{field_name} must be a string")
    try:
        return _strip_required(value)
    except ValueError as exc:
        raise AgentProjectMetadataParseError(f"{field_name} must be non-empty") from exc


def _parse_string_tuple(value: object, *, field_name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        try:
            return (_strip_required(value),)
        except ValueError as exc:
            raise AgentProjectMetadataParseError(f"{field_name} must be non-empty") from exc
    if not isinstance(value, list):
        raise AgentProjectMetadataParseError(f"{field_name} must be a list of strings")
    items: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str):
            raise AgentProjectMetadataParseError(f"{field_name}[{index}] must be a string")
        try:
            items.append(_strip_required(item))
        except ValueError as exc:
            raise AgentProjectMetadataParseError(f"{field_name}[{index}] must be non-empty") from exc
    return tuple(items)


def _nested_table(root: Mapping[str, Any], path: Sequence[str]) -> Mapping[str, Any] | None:
    current: object = root
    for index, key in enumerate(path):
        if current is None:
            return None
        mapping = _require_mapping(current, field_name=".".join(path[:index]) or "pyproject")
        current = mapping.get(key)
    if current is None:
        return None
    return _require_mapping(current, field_name=".".join(path))


def parse_agent_project_pyproject(source: str) -> AgentProjectMetadata:
    """Parse agent package metadata from a pyproject.toml document.

    Canonical capability rows live in ``[[tool.intergrax.agent.contracts]]``.
    ``agent_version`` is projected from ``[project].version`` — never synthesized.
    ``[project].dependencies`` are PEP 621 install deps and are not runtime-graph
    agent dependencies; those remain the ``dependencies`` field supplied by the
    installation metadata provider.
    """
    try:
        loaded = tomllib.loads(source)
    except tomllib.TOMLDecodeError as exc:
        raise AgentProjectMetadataParseError("invalid pyproject TOML") from exc
    root = _require_mapping(loaded, field_name="pyproject")
    project = root.get("project")
    if project is None:
        raise AgentProjectMetadataParseError("missing [project] table")
    project_mapping = _require_mapping(project, field_name="project")
    distribution_package_id = _optional_str(
        project_mapping.get("name"),
        field_name="project.name",
    )
    if distribution_package_id is None:
        raise AgentProjectMetadataParseError("missing [project].name")
    package_version = None
    raw_version = project_mapping.get("version")
    if raw_version is not None:
        package_version = _optional_str(raw_version, field_name="project.version")

    agent_table = _nested_table(root, _AGENT_TOOL_TABLE)
    declared_contracts: tuple[AgentPackageContractDeclaration, ...] = ()
    if agent_table is not None:
        raw_contracts = agent_table.get("contracts")
        if raw_contracts is None:
            declared_contracts = ()
        elif not isinstance(raw_contracts, list):
            raise AgentProjectMetadataParseError("tool.intergrax.agent.contracts must be an array")
        else:
            parsed: list[AgentPackageContractDeclaration] = []
            for index, item in enumerate(raw_contracts):
                prefix = f"tool.intergrax.agent.contracts[{index}]"
                mapping = _require_mapping(item, field_name=prefix)
                unknown = sorted(set(mapping) - {"contract_id", "capabilities", "skill_ids", "tool_ids"})
                if unknown:
                    raise AgentProjectMetadataParseError(
                        f"{prefix} unknown field(s): {', '.join(unknown)}"
                    )
                contract_id = _optional_str(mapping.get("contract_id"), field_name=f"{prefix}.contract_id")
                if contract_id is None:
                    raise AgentProjectMetadataParseError(f"{prefix} missing contract_id")
                try:
                    parsed.append(
                        AgentPackageContractDeclaration(
                            contract_id=contract_id,
                            capabilities=_parse_string_tuple(
                                mapping.get("capabilities"),
                                field_name=f"{prefix}.capabilities",
                            ),
                            skill_ids=_parse_string_tuple(
                                mapping.get("skill_ids"),
                                field_name=f"{prefix}.skill_ids",
                            ),
                            tool_ids=_parse_string_tuple(
                                mapping.get("tool_ids"),
                                field_name=f"{prefix}.tool_ids",
                            ),
                        )
                    )
                except ValueError as exc:
                    raise AgentProjectMetadataParseError(f"invalid {prefix}") from exc
            declared_contracts = tuple(parsed)

    if declared_contracts and package_version is None:
        raise AgentProjectMetadataParseError(
            "declared agent contracts require [project].version; no synthetic version fallback"
        )

    return AgentProjectMetadata(
        distribution_package_id=distribution_package_id,
        package_version=package_version,
        declared_contracts=declared_contracts,
    )


def project_agent_capability_descriptors(
    metadata: AgentProjectMetadata,
) -> tuple[AgentCapabilityDescriptor, ...]:
    """Project package-owned declarations into architecture descriptors."""
    if not metadata.declared_contracts:
        return ()
    if metadata.package_version is None:
        raise AgentProjectMetadataParseError(
            f"package {metadata.distribution_package_id!r} declares contracts but has no "
            "package_version; no synthetic version fallback"
        )
    return tuple(
        AgentCapabilityDescriptor(
            contract_id=declaration.contract_id,
            agent_version=metadata.package_version,
            capabilities=declaration.capabilities,
            skill_ids=declaration.skill_ids,
            tool_ids=declaration.tool_ids,
        )
        for declaration in metadata.declared_contracts
    )
