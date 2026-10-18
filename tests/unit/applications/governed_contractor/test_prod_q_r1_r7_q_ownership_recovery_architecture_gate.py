# © Artur Czarnecki. All rights reserved.

"""Architecture gates — reflex external-work ownership recovery (IMPL-01-R1)."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO = Path(__file__).resolve().parents[4]
_STRICT_WIRING = (
    _REPO / "applications" / "governed_contractor_application" / "host" / "strict_external_work_host_wiring.py"
)
_PORT = (
    _REPO
    / "applications"
    / "governed_contractor_application"
    / "host"
    / "sanctioned_external_work_operation_port.py"
)
_ORCHESTRATOR = _REPO / "applications" / "governed_contractor_application" / "host" / "orchestrator.py"
_FACTORY = _REPO / "applications" / "governed_contractor_application" / "host" / "factory.py"
_STORES = _REPO / "applications" / "governed_contractor_application" / "host" / "stores.py"


def test_strict_wiring_does_not_mint_per_call_in_memory_stores() -> None:
    text = _STRICT_WIRING.read_text(encoding="utf-8")
    assert "InMemoryGovernedExecutionStore()" not in text
    assert "host_stores" in text


def test_port_has_no_session_create_result_cache() -> None:
    text = _PORT.read_text(encoding="utf-8")
    assert "create_result_by_execution_id" not in text
    assert "resolve_create_adapter_result_for_follow_on" in text


def test_orchestrator_exposes_follow_on_resolver_and_record_write() -> None:
    text = _ORCHESTRATOR.read_text(encoding="utf-8")
    assert "resolve_create_adapter_result_for_follow_on" in text
    assert "put_create_follow_on_record" in text
    assert "create_redispatch_prohibited" in text


def test_governed_execution_store_declares_follow_on_protocol() -> None:
    text = _STORES.read_text(encoding="utf-8")
    assert "put_create_follow_on_record" in text
    assert "get_create_follow_on_record" in text


def test_factory_rebinds_revision_bound_external_contractor_agent() -> None:
    text = _FACTORY.read_text(encoding="utf-8")
    assert "rebind_strict_sanctioned_external_contractor_adapter_registry" in text


def test_strict_wiring_has_no_implicit_principal_defaults() -> None:
    tree = ast.parse(_STRICT_WIRING.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == (
            "build_strict_external_work_host_session_resources"
        ):
            for default in node.args.defaults:
                if isinstance(default, ast.Constant) and default.value in {
                    "default-workspace",
                    "strict-host-principal",
                }:
                    pytest.fail("implicit strict principal/workspace defaults remain")
