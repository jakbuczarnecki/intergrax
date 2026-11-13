# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Typed MCP server definition contract (``mcp-server-definition.v1``).

No FastMCP imports — composition lives in ``intergrax.applications._shared.mcp_server_engine``.
"""

from __future__ import annotations

import json
from enum import StrEnum
from typing import Annotated, Final, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from intergrax.contracts.validation import compute_sha256_content_digest

MCP_SERVER_DEFINITION_CONTRACT_VERSION: Final = "mcp-server-definition.v1"

_NON_EMPTY = Field(min_length=1)


class McpServerDefinitionErrorCode(StrEnum):
    """Stable pre-mount and exposure error codes (REQ-MCP-P-05)."""

    MCP_DEF_INVALID = "MCP_DEF_INVALID"
    MCP_BINDING_UNKNOWN = "MCP_BINDING_UNKNOWN"
    MCP_BINDING_RETIRED = "MCP_BINDING_RETIRED"
    MCP_BINDING_UNKNOWN_TOOL = "MCP_BINDING_UNKNOWN_TOOL"
    MCP_PRINCIPAL_REQUIRED = "MCP_PRINCIPAL_REQUIRED"
    MCP_TENANT_ASSERTION_CONFLICT = "MCP_TENANT_ASSERTION_CONFLICT"
    MCP_RESULT_BOUND_EXCEEDED = "MCP_RESULT_BOUND_EXCEEDED"
    # Contract v1.1 (MCP-RC, OD-MCP-RC-01 D-4): canonical wire codes for the runtime paths that previously
    # surfaced ad-hoc lowercase strings. The legacy string stays in the error detail.
    MCP_EXECUTION_BINDING_INVALID = "MCP_EXECUTION_BINDING_INVALID"
    MCP_CALLER_NOT_AUTHORIZED = "MCP_CALLER_NOT_AUTHORIZED"
    MCP_CATALOG_TOOL_REFUSED = "MCP_CATALOG_TOOL_REFUSED"
    MCP_TOOL_INPUT_INVALID = "MCP_TOOL_INPUT_INVALID"
    MCP_EXECUTION_REFUSED = "MCP_EXECUTION_REFUSED"
    MCP_INTERNAL_ERROR = "MCP_INTERNAL_ERROR"
    MCP_DEADLINE_EXCEEDED = "MCP_DEADLINE_EXCEEDED"
    MCP_CONCURRENT_CALL_LIMIT_EXCEEDED = "MCP_CONCURRENT_CALL_LIMIT_EXCEEDED"


class McpHostBuiltinId(StrEnum):
    """Built-in host MCP tools owned by the composition engine."""

    LIST_AGENTS = "host_list_agents"
    RUN_AGENT = "host_run_agent"
    LIST_CATALOG_TOOLS = "host_list_catalog_tools"
    DESCRIBE_CATALOG_TOOL = "host_describe_catalog_tool"
    INVOKE_CATALOG_TOOL = "host_invoke_catalog_tool"


class McpBindingVisibility(StrEnum):
    PUBLIC = "public"
    HIDDEN = "hidden"


class McpHostBuiltinBinding(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["host_builtin"] = "host_builtin"
    binding_id: str = _NON_EMPTY
    builtin: McpHostBuiltinId
    visibility: McpBindingVisibility = McpBindingVisibility.PUBLIC
    retired: bool = False


class McpCatalogToolBinding(BaseModel):
    """Allowlisted catalog ``tool_id`` — schema SSOT remains ``ToolContract``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["catalog_tool"] = "catalog_tool"
    binding_id: str = _NON_EMPTY
    tool_id: str = _NON_EMPTY
    visibility: McpBindingVisibility = McpBindingVisibility.PUBLIC
    retired: bool = False


class McpHostCustomBinding(BaseModel):
    """Named custom host tool (e.g. research pipeline) expressed as binding, not shadow registry."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["host_custom"] = "host_custom"
    binding_id: str = _NON_EMPTY
    tool_name: str = _NON_EMPTY
    visibility: McpBindingVisibility = McpBindingVisibility.PUBLIC
    retired: bool = False


McpCapabilityBinding = Annotated[
    McpHostBuiltinBinding | McpCatalogToolBinding | McpHostCustomBinding,
    Field(discriminator="kind"),
]


class McpExposurePolicy(BaseModel):
    """Allowlisted bindings only — dynamic change via definition recomposition (OD-MCP-04-A)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool = True
    allowed_binding_ids: tuple[str, ...] = ()

    @field_validator("allowed_binding_ids")
    @classmethod
    def _strip_allowlist(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(item.strip() for item in value if item and item.strip())


class McpServerDefinition(BaseModel):
    """Declarative MCP server shape for one application mount."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    server_id: str = _NON_EMPTY
    display_name: str = _NON_EMPTY
    contract_version: Literal["mcp-server-definition.v1"] = MCP_SERVER_DEFINITION_CONTRACT_VERSION
    bindings: tuple[McpCapabilityBinding, ...] = ()
    exposure_policy: McpExposurePolicy = Field(default_factory=McpExposurePolicy)
    default_capability: str = _NON_EMPTY
    default_tenant_id: str = "mcp"
    default_user_id: str = "mcp-user"

    @model_validator(mode="after")
    def _binding_ids_unique(self) -> McpServerDefinition:
        seen: set[str] = set()
        for binding in self.bindings:
            if binding.binding_id in seen:
                raise ValueError(f"duplicate binding_id: {binding.binding_id}")
            seen.add(binding.binding_id)
        return self


class McpServerDefinitionValidationError(ValueError):
    """Definition or exposure validation failed before server composition."""

    def __init__(self, code: McpServerDefinitionErrorCode, message: str) -> None:
        super().__init__(message)
        self.code = code


def build_mcp_server_definition(
    *,
    server_id: str,
    display_name: str,
    default_capability: str,
    bindings: tuple[McpCapabilityBinding, ...],
    exposure_policy: McpExposurePolicy | None = None,
    default_tenant_id: str = "mcp",
    default_user_id: str = "mcp-user",
) -> McpServerDefinition:
    """Programmatic builder — same semantics as declarative ``McpServerDefinition`` construction."""
    policy = exposure_policy
    if policy is None:
        policy = McpExposurePolicy(
            enabled=True,
            allowed_binding_ids=tuple(binding.binding_id for binding in bindings),
        )
    return McpServerDefinition(
        server_id=server_id.strip(),
        display_name=display_name.strip(),
        default_capability=default_capability.strip(),
        bindings=bindings,
        exposure_policy=policy,
        default_tenant_id=default_tenant_id.strip(),
        default_user_id=default_user_id.strip(),
    )


def iter_definition_binding_ids(definition: McpServerDefinition) -> frozenset[str]:
    return frozenset(binding.binding_id for binding in definition.bindings)


def compute_mcp_exposure_schema_fingerprint(definition: McpServerDefinition) -> str:
    """Deterministic fingerprint of contract version + exposed binding shapes (REQ-MCP-C-01 / MCP-P4-03)."""
    exposed = resolve_effective_bindings(definition)
    payload = {
        "contract_version": definition.contract_version,
        "bindings": [binding.model_dump(mode="json") for binding in exposed],
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return compute_sha256_content_digest(canonical)


def resolve_binding_for_invocation(
    definition: McpServerDefinition,
    binding_id: str,
) -> McpCapabilityBinding:
    """Resolve a binding by id for invocation; fail closed on retired or non-exposed bindings."""
    target = binding_id.strip()
    if not target:
        raise McpServerDefinitionValidationError(
            McpServerDefinitionErrorCode.MCP_DEF_INVALID,
            "binding_id is required",
        )
    binding_map = {binding.binding_id: binding for binding in definition.bindings}
    binding = binding_map.get(target)
    if binding is None:
        raise McpServerDefinitionValidationError(
            McpServerDefinitionErrorCode.MCP_BINDING_UNKNOWN,
            f"unknown binding_id: {target}",
        )
    if binding.retired:
        raise McpServerDefinitionValidationError(
            McpServerDefinitionErrorCode.MCP_BINDING_RETIRED,
            f"binding retired: {target}",
        )
    exposed_ids = frozenset(item.binding_id for item in resolve_effective_bindings(definition))
    if target not in exposed_ids:
        raise McpServerDefinitionValidationError(
            McpServerDefinitionErrorCode.MCP_BINDING_UNKNOWN,
            f"binding not exposed: {target}",
        )
    return binding


def resolve_effective_bindings(definition: McpServerDefinition) -> tuple[McpCapabilityBinding, ...]:
    """Apply exposure policy allowlist and enabled flag (discovery visibility)."""
    if not definition.exposure_policy.enabled:
        return ()
    allowed = frozenset(definition.exposure_policy.allowed_binding_ids)
    if not allowed:
        return ()
    exposed: list[McpCapabilityBinding] = []
    for binding in definition.bindings:
        if binding.binding_id not in allowed:
            continue
        if binding.retired or binding.visibility == McpBindingVisibility.HIDDEN:
            continue
        exposed.append(binding)
    return tuple(exposed)


def validate_mcp_server_definition(
    definition: McpServerDefinition,
    *,
    known_catalog_tool_ids: frozenset[str] | None = None,
) -> None:
    """Fail-closed validation before FastMCP composition."""
    if definition.contract_version != MCP_SERVER_DEFINITION_CONTRACT_VERSION:
        raise McpServerDefinitionValidationError(
            McpServerDefinitionErrorCode.MCP_DEF_INVALID,
            f"unsupported contract_version: {definition.contract_version!r}",
        )
    if not definition.server_id.strip() or not definition.display_name.strip():
        raise McpServerDefinitionValidationError(
            McpServerDefinitionErrorCode.MCP_DEF_INVALID,
            "server_id and display_name are required",
        )
    if not definition.default_capability.strip():
        raise McpServerDefinitionValidationError(
            McpServerDefinitionErrorCode.MCP_DEF_INVALID,
            "default_capability is required",
        )

    binding_ids = iter_definition_binding_ids(definition)
    for allowed_id in definition.exposure_policy.allowed_binding_ids:
        if allowed_id not in binding_ids:
            raise McpServerDefinitionValidationError(
                McpServerDefinitionErrorCode.MCP_BINDING_UNKNOWN,
                f"exposure allowlist references unknown binding_id: {allowed_id}",
            )

    catalog_ids: set[str] = set()
    for binding in definition.bindings:
        if isinstance(binding, McpCatalogToolBinding):
            catalog_ids.add(binding.tool_id)
            if known_catalog_tool_ids is not None and binding.tool_id not in known_catalog_tool_ids:
                raise McpServerDefinitionValidationError(
                    McpServerDefinitionErrorCode.MCP_BINDING_UNKNOWN_TOOL,
                    f"unknown catalog tool_id in binding: {binding.tool_id}",
                )

    if catalog_ids and known_catalog_tool_ids is not None:
        exposed = resolve_effective_bindings(definition)
        catalog_surface = {
            McpHostBuiltinId.LIST_CATALOG_TOOLS,
            McpHostBuiltinId.DESCRIBE_CATALOG_TOOL,
            McpHostBuiltinId.INVOKE_CATALOG_TOOL,
        }
        has_surface = any(
            isinstance(item, McpHostBuiltinBinding) and item.builtin in catalog_surface for item in exposed
        )
        if not has_surface:
            raise McpServerDefinitionValidationError(
                McpServerDefinitionErrorCode.MCP_DEF_INVALID,
                "catalog_tool bindings require catalog introspection host builtins in exposure",
            )


__all__ = [
    "MCP_SERVER_DEFINITION_CONTRACT_VERSION",
    "McpBindingVisibility",
    "McpCapabilityBinding",
    "McpCatalogToolBinding",
    "McpExposurePolicy",
    "McpHostBuiltinBinding",
    "McpHostBuiltinId",
    "McpHostCustomBinding",
    "McpServerDefinition",
    "McpServerDefinitionErrorCode",
    "McpServerDefinitionValidationError",
    "build_mcp_server_definition",
    "compute_mcp_exposure_schema_fingerprint",
    "iter_definition_binding_ids",
    "resolve_binding_for_invocation",
    "resolve_effective_bindings",
    "validate_mcp_server_definition",
]
