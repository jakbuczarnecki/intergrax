# © Artur Czarnecki. All rights reserved.

"""Architecture gates — IMPL-01-R2 strict composition recovery."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO = Path(__file__).resolve().parents[4]
_STRICT_WIRING = (
    _REPO
    / "applications"
    / "governed_contractor_application"
    / "host"
    / "strict_external_work_host_wiring.py"
)
_FACTORY = _REPO / "applications" / "governed_contractor_application" / "host" / "factory.py"
_SHARED_COMPOSITION = (
    _REPO / "intergrax" / "applications" / "_shared" / "production_process_composition.py"
)
_REBIND = (
    _REPO
    / "applications"
    / "governed_contractor_application"
    / "host"
    / "strict_revision_bound_agent_registry_rebind.py"
)


def test_strict_wiring_requires_explicit_host_stores_no_in_memory_fallback() -> None:
    text = _STRICT_WIRING.read_text(encoding="utf-8")
    assert "build_process_scoped_governed_external_work_host_stores" not in text
    assert "quote_first_partner_capability_fixture" not in text
    assert "InMemoryGovernedExecutionStore" not in text
    tree = ast.parse(text)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == (
            "build_strict_external_work_host_session_resources"
        ):
            for default in node.args.defaults:
                if isinstance(default, ast.Constant) and default.value == "gec3_deterministic_fake":
                    pytest.fail("default fake provider_id remains on strict wiring")


def test_factory_has_no_registry_overlay_or_rebind() -> None:
    text = _FACTORY.read_text(encoding="utf-8")
    assert "strict_revision_bound_agent_registry_rebind" not in text
    assert "rebind_strict_sanctioned_external_contractor_adapter_registry" not in text
    assert "quote_first_partner_capability_fixture" not in text
    assert "reproject_registry_with_strict_sanctioned_external_work" in text
    assert "governed_external_work_host_store_bundle" in text


def test_registry_overlay_module_removed() -> None:
    assert not _REBIND.exists()


def test_shared_composition_declares_gca_store_bundle_port() -> None:
    text = _SHARED_COMPOSITION.read_text(encoding="utf-8")
    assert "governed_external_work_host_store_bundle" in text
    assert "GovernedExternalWorkHostStoreBundlePort" in text
    assert "governed_contractor_application" not in text


def test_host_overriding_resolver_delegate_first_no_host_factory_short_circuit() -> None:
    path = (
        _REPO
        / "intergrax"
        / "applications"
        / "_shared"
        / "host_overriding_runtime_agent_factory_resolver.py"
    )
    text = path.read_text(encoding="utf-8")
    assert "host_factory" not in text
    assert "wrap_authoritative_factory" in text
    assert "self._delegate.resolve_factory" in text


def test_strict_wiring_requires_capabilities_argument() -> None:
    tree = ast.parse(_STRICT_WIRING.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == (
            "build_strict_external_work_host_session_resources"
        ):
            arg_names = [a.arg for a in node.args.args]
            arg_names.extend(a.arg for a in node.args.kwonlyargs)
            assert "host_stores" in arg_names
            assert "capabilities" in arg_names
            assert node.args.defaults == []
            assert all(default is None for default in node.args.kw_defaults)
