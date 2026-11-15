# © Artur Czarnecki. All rights reserved.

"""MCP-SERVER-X P7 lane L1: typed definition contract qualification.

Level: **L1 (unit)**. No server is composed, no transport is opened; this module is
neither an L2 (in-process) nor an L3 (physical streamable HTTP) test and must never be
counted as one.

Covers (ticket T-IGX-035-H, ``MCP_SERVER_X_QUALIFICATION_MATRIX.md``):

- MCP-ACC-02: typed server definition (error code catalog vs the contract and protocol
  matrix, incomplete definitions rejected, contract free of FastMCP).
- MCP-ACC-04 (L1 part): the exposure policy filters bindings.
- MCP-ACC-16 (L1 part): contract version constant, exposure schema fingerprint, COMPAT
  inventory.

Complements ``tests/unit/contracts/test_mcp_server_definition.py`` without repeating it.
"""

from __future__ import annotations

import ast
import importlib
import re
import subprocess
import sys
from pathlib import Path

import pytest
from intergrax.contracts.mcp_server_definition import (
    MCP_SERVER_DEFINITION_CONTRACT_VERSION,
    McpBindingVisibility,
    McpCatalogToolBinding,
    McpExposurePolicy,
    McpHostBuiltinBinding,
    McpHostBuiltinId,
    McpHostCustomBinding,
    McpServerDefinition,
    McpServerDefinitionErrorCode,
    McpServerDefinitionValidationError,
    build_mcp_server_definition,
    compute_mcp_exposure_schema_fingerprint,
    iter_definition_binding_ids,
    resolve_binding_for_invocation,
    resolve_effective_bindings,
    validate_mcp_server_definition,
)
from pydantic import ValidationError

pytestmark = [pytest.mark.unit]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_MATRIX_PATH = (
    _REPO_ROOT
    / "docs"
    / "project"
    / "maintainers"
    / "qualification"
    / "mcp_server_x"
    / "MCP_SERVER_X_CONTRACT_AND_PROTOCOL_MATRIX.md"
)

_V1_ERROR_CODES = frozenset(
    {
        "MCP_DEF_INVALID",
        "MCP_BINDING_UNKNOWN",
        "MCP_BINDING_RETIRED",
        "MCP_BINDING_UNKNOWN_TOOL",
        "MCP_PRINCIPAL_REQUIRED",
        "MCP_TENANT_ASSERTION_CONFLICT",
        "MCP_RESULT_BOUND_EXCEEDED",
    }
)
# MCP-RC contract v1.1 (OD-MCP-RC-01 D-4): documented in MCP_SERVER_X_RC_CONTRACT_MATRIX.md.
_V1_1_ADDED_ERROR_CODES = frozenset(
    {
        "MCP_EXECUTION_BINDING_INVALID",
        "MCP_CALLER_NOT_AUTHORIZED",
        "MCP_CATALOG_TOOL_REFUSED",
        "MCP_TOOL_INPUT_INVALID",
        "MCP_EXECUTION_REFUSED",
        "MCP_INTERNAL_ERROR",
        "MCP_DEADLINE_EXCEEDED",
        "MCP_CONCURRENT_CALL_LIMIT_EXCEEDED",
    }
)
_EXPECTED_ERROR_CODES = _V1_ERROR_CODES | _V1_1_ADDED_ERROR_CODES

_RC_MATRIX_PATH = _MATRIX_PATH.with_name("MCP_SERVER_X_RC_CONTRACT_MATRIX.md")


def _list_agents(binding_id: str = "list_agents", **kwargs: object) -> McpHostBuiltinBinding:
    return McpHostBuiltinBinding(binding_id=binding_id, builtin=McpHostBuiltinId.LIST_AGENTS, **kwargs)  # type: ignore[arg-type]


def _run_agent(binding_id: str = "run_agent", **kwargs: object) -> McpHostBuiltinBinding:
    return McpHostBuiltinBinding(binding_id=binding_id, builtin=McpHostBuiltinId.RUN_AGENT, **kwargs)  # type: ignore[arg-type]


def _definition(
    *bindings: McpHostBuiltinBinding | McpCatalogToolBinding | McpHostCustomBinding,
    exposure_policy: McpExposurePolicy | None = None,
) -> McpServerDefinition:
    return build_mcp_server_definition(
        server_id="srv",
        display_name="Srv",
        default_capability="cap.a",
        bindings=tuple(bindings),
        exposure_policy=exposure_policy,
    )


def _matrix_text() -> str:
    return _MATRIX_PATH.read_text(encoding="utf-8")


def _matrix_error_codes() -> set[str]:
    """Codes listed in the ``## Error codes`` table of the contract and protocol matrix."""
    codes: set[str] = set()
    for text in (_matrix_text(), _RC_MATRIX_PATH.read_text(encoding="utf-8")):
        section = text.split("## Error codes", 1)[1].split("\n## ", 1)[0]
        codes |= set(re.findall(r"^\|\s*`(MCP_[A-Z_]+)`\s*\|", section, flags=re.MULTILINE))
    return codes


# --------------------------------------------------------------------------- MCP-ACC-02


def test_acc_02_error_code_enum_has_exactly_the_v1_and_v1_1_stable_codes() -> None:
    values = {member.value for member in McpServerDefinitionErrorCode}
    assert values == _EXPECTED_ERROR_CODES
    assert len(list(McpServerDefinitionErrorCode)) == 15
    for member in McpServerDefinitionErrorCode:
        assert member.name == member.value
        assert str(member) == member.value


def test_acc_02_matrix_error_codes_are_all_defined_in_enum() -> None:
    documented = _matrix_error_codes()
    assert documented, "matrix Error codes table could not be parsed"
    enum_values = {member.value for member in McpServerDefinitionErrorCode}
    assert documented <= enum_values, f"matrix codes missing in enum: {sorted(documented - enum_values)}"


def test_acc_02_matrix_error_codes_table_lists_every_enum_code() -> None:
    """The P7 matrix and the RC v1.1 matrix together list every enum code."""
    enum_values = {member.value for member in McpServerDefinitionErrorCode}
    assert enum_values - _matrix_error_codes() == set()


def test_acc_02_matrix_pins_definition_contract_version() -> None:
    assert MCP_SERVER_DEFINITION_CONTRACT_VERSION == "mcp-server-definition.v1"
    assert f"`{MCP_SERVER_DEFINITION_CONTRACT_VERSION}`" in _matrix_text()


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({"server_id": "", "display_name": "x", "default_capability": "c"}, id="empty_server_id"),
        pytest.param({"server_id": "s", "display_name": "", "default_capability": "c"}, id="empty_display_name"),
        pytest.param({"server_id": "s", "display_name": "x", "default_capability": ""}, id="empty_default_capability"),
        pytest.param({"display_name": "x", "default_capability": "c"}, id="missing_server_id"),
        pytest.param({"server_id": "s", "default_capability": "c"}, id="missing_display_name"),
        pytest.param({"server_id": "s", "display_name": "x"}, id="missing_default_capability"),
        pytest.param(
            {"server_id": "s", "display_name": "x", "default_capability": "c", "unknown_field": 1},
            id="unknown_field_forbidden",
        ),
        pytest.param(
            {
                "server_id": "s",
                "display_name": "x",
                "default_capability": "c",
                "contract_version": "mcp-server-definition.v0",
            },
            id="foreign_contract_version",
        ),
    ],
)
def test_acc_02_incomplete_or_foreign_definition_rejected_at_construction(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        McpServerDefinition(**kwargs)  # type: ignore[arg-type]


def test_acc_02_duplicate_binding_id_rejected() -> None:
    with pytest.raises(ValidationError, match="duplicate binding_id"):
        _definition(_list_agents("same"), _run_agent("same"))


def test_acc_02_binding_with_empty_id_rejected() -> None:
    with pytest.raises(ValidationError):
        McpHostBuiltinBinding(binding_id="", builtin=McpHostBuiltinId.LIST_AGENTS)
    with pytest.raises(ValidationError):
        McpCatalogToolBinding(binding_id="b", tool_id="")
    with pytest.raises(ValidationError):
        McpHostCustomBinding(binding_id="b", tool_name="")


def test_acc_02_definition_is_frozen() -> None:
    definition = _definition(_list_agents())
    with pytest.raises(ValidationError):
        definition.server_id = "other"  # type: ignore[misc]


@pytest.mark.parametrize(
    "field",
    ["server_id", "display_name", "default_capability"],
)
def test_acc_02_blank_required_field_fails_validation_with_def_invalid(field: str) -> None:
    """Whitespace-only values pass pydantic ``min_length`` but must fail ``validate_mcp_server_definition``."""
    base = {"server_id": "s", "display_name": "x", "default_capability": "c"}
    base[field] = "   "
    definition = McpServerDefinition(**base)
    with pytest.raises(McpServerDefinitionValidationError) as exc:
        validate_mcp_server_definition(definition)
    assert exc.value.code is McpServerDefinitionErrorCode.MCP_DEF_INVALID


def test_acc_02_unsupported_contract_version_fails_validation_with_def_invalid() -> None:
    definition = McpServerDefinition.model_construct(
        server_id="s",
        display_name="x",
        contract_version="mcp-server-definition.v0",
        bindings=(),
        exposure_policy=McpExposurePolicy(),
        default_capability="c",
        default_tenant_id="mcp",
        default_user_id="mcp-user",
    )
    with pytest.raises(McpServerDefinitionValidationError) as exc:
        validate_mcp_server_definition(definition)
    assert exc.value.code is McpServerDefinitionErrorCode.MCP_DEF_INVALID
    assert "unsupported contract_version" in str(exc.value)


def test_acc_02_unknown_allowlist_binding_reports_the_offending_id() -> None:
    definition = _definition(
        _list_agents(),
        exposure_policy=McpExposurePolicy(enabled=True, allowed_binding_ids=("list_agents", "ghost")),
    )
    with pytest.raises(McpServerDefinitionValidationError) as exc:
        validate_mcp_server_definition(definition)
    assert exc.value.code is McpServerDefinitionErrorCode.MCP_BINDING_UNKNOWN
    assert "ghost" in str(exc.value)


def test_acc_02_unknown_catalog_tool_reports_the_offending_tool_id() -> None:
    definition = _definition(
        McpHostBuiltinBinding(binding_id="lct", builtin=McpHostBuiltinId.LIST_CATALOG_TOOLS),
        McpCatalogToolBinding(binding_id="t1", tool_id="does.not.exist"),
    )
    with pytest.raises(McpServerDefinitionValidationError) as exc:
        validate_mcp_server_definition(definition, known_catalog_tool_ids=frozenset({"other.tool"}))
    assert exc.value.code is McpServerDefinitionErrorCode.MCP_BINDING_UNKNOWN_TOOL
    assert "does.not.exist" in str(exc.value)


def test_acc_02_catalog_binding_without_catalog_builtin_surface_fails_with_def_invalid() -> None:
    definition = _definition(
        _list_agents(),
        McpCatalogToolBinding(binding_id="t1", tool_id="known.tool"),
    )
    with pytest.raises(McpServerDefinitionValidationError) as exc:
        validate_mcp_server_definition(definition, known_catalog_tool_ids=frozenset({"known.tool"}))
    assert exc.value.code is McpServerDefinitionErrorCode.MCP_DEF_INVALID
    assert "catalog introspection" in str(exc.value)


def test_acc_02_valid_definition_with_catalog_surface_passes_validation() -> None:
    definition = _definition(
        McpHostBuiltinBinding(binding_id="lct", builtin=McpHostBuiltinId.LIST_CATALOG_TOOLS),
        McpCatalogToolBinding(binding_id="t1", tool_id="known.tool"),
    )
    assert validate_mcp_server_definition(definition, known_catalog_tool_ids=frozenset({"known.tool"})) is None


def test_acc_02_resolve_binding_error_codes_unknown_retired_unexposed_blank() -> None:
    definition = _definition(
        _list_agents(),
        _run_agent(retired=True),
        McpHostCustomBinding(binding_id="custom", tool_name="extra"),
        exposure_policy=McpExposurePolicy(enabled=True, allowed_binding_ids=("list_agents", "run_agent")),
    )
    expectations = {
        "missing": McpServerDefinitionErrorCode.MCP_BINDING_UNKNOWN,
        "run_agent": McpServerDefinitionErrorCode.MCP_BINDING_RETIRED,
        "custom": McpServerDefinitionErrorCode.MCP_BINDING_UNKNOWN,  # defined but not on the allowlist
        "   ": McpServerDefinitionErrorCode.MCP_DEF_INVALID,
    }
    for binding_id, code in expectations.items():
        with pytest.raises(McpServerDefinitionValidationError) as exc:
            resolve_binding_for_invocation(definition, binding_id)
        assert exc.value.code is code, binding_id
    assert resolve_binding_for_invocation(definition, " list_agents ").binding_id == "list_agents"


def test_acc_02_validation_error_is_value_error_carrying_code() -> None:
    err = McpServerDefinitionValidationError(McpServerDefinitionErrorCode.MCP_DEF_INVALID, "boom")
    assert isinstance(err, ValueError)
    assert err.code is McpServerDefinitionErrorCode.MCP_DEF_INVALID
    assert str(err) == "boom"


def test_acc_02_contract_module_imports_no_fastmcp_directly_or_transitively() -> None:
    path = Path(importlib.import_module("intergrax.contracts.mcp_server_definition").__file__ or "")
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".")[0])
    assert roots <= {"__future__", "json", "enum", "typing", "pydantic", "intergrax"}, sorted(roots)
    assert not any("fastmcp" in name or name == "mcp" for name in roots)

    # Transitive check in a clean interpreter: importing the contract must not load FastMCP / mcp SDK.
    probe = (
        "import sys; import intergrax.contracts.mcp_server_definition; "
        "bad = sorted(m for m in sys.modules if m == 'fastmcp' or m.startswith('fastmcp.') "
        "or m == 'mcp' or m.startswith('mcp.')); "
        "print('LOADED:' + ','.join(bad))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip().splitlines()[-1] == "LOADED:"


# --------------------------------------------------------------------------- MCP-ACC-04 (L1)


def test_acc_04_disabled_policy_exposes_nothing() -> None:
    definition = _definition(
        _list_agents(),
        _run_agent(),
        exposure_policy=McpExposurePolicy(enabled=False, allowed_binding_ids=("list_agents", "run_agent")),
    )
    assert resolve_effective_bindings(definition) == ()
    with pytest.raises(McpServerDefinitionValidationError) as exc:
        resolve_binding_for_invocation(definition, "list_agents")
    assert exc.value.code is McpServerDefinitionErrorCode.MCP_BINDING_UNKNOWN


def test_acc_04_empty_allowlist_exposes_nothing_fail_closed() -> None:
    definition = _definition(
        _list_agents(),
        exposure_policy=McpExposurePolicy(enabled=True, allowed_binding_ids=()),
    )
    assert resolve_effective_bindings(definition) == ()


def test_acc_04_binding_not_on_allowlist_is_dropped_and_not_invocable() -> None:
    definition = _definition(
        _list_agents(),
        _run_agent(),
        exposure_policy=McpExposurePolicy(enabled=True, allowed_binding_ids=("list_agents",)),
    )
    assert [b.binding_id for b in resolve_effective_bindings(definition)] == ["list_agents"]
    with pytest.raises(McpServerDefinitionValidationError) as exc:
        resolve_binding_for_invocation(definition, "run_agent")
    assert exc.value.code is McpServerDefinitionErrorCode.MCP_BINDING_UNKNOWN
    assert "not exposed" in str(exc.value)


def test_acc_04_hidden_and_retired_bindings_never_exposed_even_when_allowlisted() -> None:
    definition = _definition(
        _list_agents(),
        _run_agent(visibility=McpBindingVisibility.HIDDEN),
        McpHostCustomBinding(binding_id="old", tool_name="old_tool", retired=True),
        exposure_policy=McpExposurePolicy(enabled=True, allowed_binding_ids=("list_agents", "run_agent", "old")),
    )
    assert [b.binding_id for b in resolve_effective_bindings(definition)] == ["list_agents"]


def test_acc_04_exposure_preserves_definition_order_not_allowlist_order() -> None:
    definition = _definition(
        _list_agents("a"),
        _run_agent("b"),
        McpHostCustomBinding(binding_id="c", tool_name="tool_c"),
        exposure_policy=McpExposurePolicy(enabled=True, allowed_binding_ids=("c", "a", "b")),
    )
    assert [b.binding_id for b in resolve_effective_bindings(definition)] == ["a", "b", "c"]


def test_acc_04_allowlist_entries_are_stripped_and_blank_entries_dropped() -> None:
    policy = McpExposurePolicy(enabled=True, allowed_binding_ids=(" list_agents ", "", "   "))
    assert policy.allowed_binding_ids == ("list_agents",)
    definition = _definition(_list_agents(), _run_agent(), exposure_policy=policy)
    assert [b.binding_id for b in resolve_effective_bindings(definition)] == ["list_agents"]


def test_acc_04_default_policy_from_builder_exposes_all_visible_active_bindings() -> None:
    definition = _definition(
        _list_agents(), _run_agent(retired=True), McpHostCustomBinding(binding_id="c", tool_name="t")
    )
    assert definition.exposure_policy.allowed_binding_ids == ("list_agents", "run_agent", "c")
    assert [b.binding_id for b in resolve_effective_bindings(definition)] == ["list_agents", "c"]


def test_acc_04_all_binding_kinds_are_filtered_by_the_same_policy() -> None:
    definition = _definition(
        McpHostBuiltinBinding(binding_id="lct", builtin=McpHostBuiltinId.LIST_CATALOG_TOOLS),
        McpCatalogToolBinding(binding_id="t1", tool_id="known.tool"),
        McpCatalogToolBinding(binding_id="t2", tool_id="other.tool"),
        McpHostCustomBinding(binding_id="c", tool_name="t"),
        exposure_policy=McpExposurePolicy(enabled=True, allowed_binding_ids=("lct", "t2")),
    )
    exposed = resolve_effective_bindings(definition)
    assert [(b.kind, b.binding_id) for b in exposed] == [("host_builtin", "lct"), ("catalog_tool", "t2")]
    assert iter_definition_binding_ids(definition) == frozenset({"lct", "t1", "t2", "c"})


# --------------------------------------------------------------------------- MCP-ACC-16 (L1)


def test_acc_16_contract_version_constant_is_pinned_and_default_on_definitions() -> None:
    assert MCP_SERVER_DEFINITION_CONTRACT_VERSION == "mcp-server-definition.v1"
    assert _definition().contract_version == MCP_SERVER_DEFINITION_CONTRACT_VERSION
    assert McpServerDefinition(server_id="s", display_name="x", default_capability="c").contract_version == (
        MCP_SERVER_DEFINITION_CONTRACT_VERSION
    )


def test_acc_16_fingerprint_is_golden_stable_sha256() -> None:
    """Golden value pins the canonical-JSON + sha256 algorithm; any accidental drift is visible."""
    fingerprint = compute_mcp_exposure_schema_fingerprint(_definition(_list_agents()))
    assert re.fullmatch(r"sha256:[0-9a-f]{64}", fingerprint)
    assert fingerprint == "sha256:9fcfb3607dada869c6529e49a307b1de2a71f9fe6bf21a9f82430e268d84fa3f"
    assert fingerprint == compute_mcp_exposure_schema_fingerprint(_definition(_list_agents()))


def test_acc_16_fingerprint_ignores_non_exposed_bindings_and_server_identity() -> None:
    base = compute_mcp_exposure_schema_fingerprint(_definition(_list_agents()))
    with_hidden = _definition(
        _list_agents(),
        _run_agent(visibility=McpBindingVisibility.HIDDEN),
    )
    with_retired = _definition(_list_agents(), _run_agent(retired=True))
    with_unlisted = _definition(
        _list_agents(),
        _run_agent(),
        exposure_policy=McpExposurePolicy(enabled=True, allowed_binding_ids=("list_agents",)),
    )
    renamed = build_mcp_server_definition(
        server_id="other",
        display_name="Other",
        default_capability="cap.z",
        bindings=(_list_agents(),),
    )
    for variant in (with_hidden, with_retired, with_unlisted, renamed):
        assert compute_mcp_exposure_schema_fingerprint(variant) == base


def test_acc_16_fingerprint_changes_when_a_binding_is_retired() -> None:
    active = _definition(_list_agents(), _run_agent())
    retired = _definition(_list_agents(), _run_agent(retired=True))
    assert compute_mcp_exposure_schema_fingerprint(active) != compute_mcp_exposure_schema_fingerprint(retired)


def test_acc_16_fingerprint_sensitive_to_binding_shape_visibility_and_order() -> None:
    fp = compute_mcp_exposure_schema_fingerprint
    base = _definition(McpCatalogToolBinding(binding_id="t", tool_id="tool.one"))
    other_tool = _definition(McpCatalogToolBinding(binding_id="t", tool_id="tool.two"))
    other_id = _definition(McpCatalogToolBinding(binding_id="t2", tool_id="tool.one"))
    other_kind = _definition(McpHostCustomBinding(binding_id="t", tool_name="tool.one"))
    assert len({fp(base), fp(other_tool), fp(other_id), fp(other_kind)}) == 4

    ab = _definition(_list_agents("a"), _run_agent("b"))
    ba = _definition(_run_agent("b"), _list_agents("a"))
    assert fp(ab) != fp(ba)


def test_acc_16_fingerprint_of_disabled_policy_equals_that_of_empty_definition() -> None:
    disabled = _definition(
        _list_agents(),
        exposure_policy=McpExposurePolicy(enabled=False, allowed_binding_ids=("list_agents",)),
    )
    empty = _definition()
    assert compute_mcp_exposure_schema_fingerprint(disabled) == compute_mcp_exposure_schema_fingerprint(empty)


def test_acc_16_compat_inventory_lists_definition_version_surfaces() -> None:
    from tests.qualification.compat_x._compat_x_inventory import COMPAT_X_INVENTORY

    rows = [
        row for row in COMPAT_X_INVENTORY if row.owner_module_path == "intergrax/contracts/mcp_server_definition.py"
    ]
    surface_ids = {row.surface_id for row in rows}
    assert any(sid.endswith(":MCP_SERVER_DEFINITION_CONTRACT_VERSION") for sid in surface_ids), surface_ids
    assert any(sid.endswith(":McpServerDefinition:contract_version") for sid in surface_ids), surface_ids
    assert {row.current_version for row in rows} == {MCP_SERVER_DEFINITION_CONTRACT_VERSION}
