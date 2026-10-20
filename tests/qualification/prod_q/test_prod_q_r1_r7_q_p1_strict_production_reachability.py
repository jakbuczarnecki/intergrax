# © Artur Czarnecki. All rights reserved.

"""PROD-Q-R1-R7-Q-P1 — strict production reachability regression gate."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from tests.qualification.prod_q._prod_q_r1_r7_q_p1_strict_reachability import (
    BOUNDARY_MODULE_RELPATHS,
    PROFILE_V1_STRICT_STARTUP_OPENER,
    PROFILE_V1_STRICT_STARTUP_RELPATH,
    STRICT_PRODUCTION_ROOTS,
    analyze_asgi_admission_snippet,
    analyze_gca_deployment_snippet,
    analyze_module_snippet,
    analyze_profile_v1_startup_snippet,
    analyze_strict_production_reachability,
    assert_all_strict_production_roots_located,
    assert_profile_v1_startup_chain_opener,
    assert_startup_chain_opener,
    assert_strict_asgi_admission_guard,
    boundary_module_paths,
    production_lifecycle_rejection_import_is_guard_only,
    repo_root,
)

pytestmark = [pytest.mark.qualification, pytest.mark.gate]

_EXPECTED_REACHABLE_FROM_PROFILE_V1_OPENER = frozenset(
    {
        (
            PROFILE_V1_STRICT_STARTUP_RELPATH,
            PROFILE_V1_STRICT_STARTUP_OPENER,
        ),
        (
            "intergrax/applications/_shared/single_node_production_profile_v1.py",
            "load_validate_and_admit_single_node_production_profile_v1_from_env",
        ),
        (
            "applications/governed_contractor_application/host/strict_gca_production_process_composition.py",
            "open_gca_strict_production_serving_session",
        ),
        (
            "intergrax/applications/_shared/strict_production_serving_launcher.py",
            "open_strict_production_serving_session",
        ),
        (
            "intergrax/applications/_shared/production_process_lifecycle.py",
            "open_production_process_lifecycle",
        ),
    }
)

_EXPECTED_REACHABLE_FROM_STRICT_OPENER = frozenset(
    {
        (
            "intergrax/applications/_shared/strict_production_serving_launcher.py",
            "open_strict_production_serving_session",
        ),
        (
            "intergrax/applications/_shared/production_process_lifecycle.py",
            "open_production_process_lifecycle",
        ),
        (
            "intergrax/applications/_shared/production_lifecycle.py",
            "reject_reference_runtime_deployment_adapter",
        ),
        (
            "intergrax/applications/_shared/production_lifecycle.py",
            "wire_production_lifecycle_services",
        ),
    }
)


def test_p1_g01_one_strict_production_root_chain() -> None:
    assert_all_strict_production_roots_located()
    assert_startup_chain_opener()
    assert_profile_v1_startup_chain_opener()
    report = analyze_strict_production_reachability()
    missing = _EXPECTED_REACHABLE_FROM_STRICT_OPENER - report.reachable_functions
    assert not missing, f"strict opener chain missing reachable functions: {missing}"
    missing_pv1 = _EXPECTED_REACHABLE_FROM_PROFILE_V1_OPENER - report.reachable_functions
    assert not missing_pv1, f"profile v1 opener chain missing reachable functions: {missing_pv1}"


def test_p1_g02_no_prohibited_execution_on_strict_reachable_graph() -> None:
    report = analyze_strict_production_reachability()
    assert not report.prohibited, "\n".join(
        f"{v.module_relpath}:{v.lineno} {v.qualname} -> {v.detail}" for v in report.prohibited
    )


def test_p1_g03_reference_lifecycle_import_without_execution_in_strict_modules() -> None:
    rel = "intergrax/applications/_shared/strict_production_serving_launcher.py"
    path = repo_root() / rel
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imports = {
        alias.asname or alias.name
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    text = path.read_text(encoding="utf-8")
    assert "open_reference_production_process_lifecycle" not in text
    report = analyze_strict_production_reachability()
    ref_calls = [v for v in report.prohibited if "open_reference" in v.detail]
    assert not ref_calls


def test_p1_g04_gca_strict_factory_does_not_construct_deployment_adapters() -> None:
    report = analyze_strict_production_reachability()
    assert not report.gca_deployment_constructions


def test_p1_g05_strict_asgi_admission_guard_structural() -> None:
    assert_strict_asgi_admission_guard()
    main = repo_root() / "applications/governed_contractor_application/host/main.py"
    main_text = main.read_text(encoding="utf-8")
    assert "StrictProductionAsgiPlaceholder" in main_text


def test_p1_g06_closed_world_inventory_complete() -> None:
    missing = [rel for rel, path in boundary_module_paths().items() if not path.is_file()]
    assert not missing, f"boundary module missing: {missing}"
    assert len(BOUNDARY_MODULE_RELPATHS) == len(boundary_module_paths())


def test_p1_production_lifecycle_fake_import_is_rejection_guard_only() -> None:
    production_lifecycle_rejection_import_is_guard_only()
    proc = repo_root() / "intergrax/applications/_shared/production_process_lifecycle.py"
    assert "FakeInMemoryRuntimeDeploymentAdapter" not in proc.read_text(encoding="utf-8")


def test_p1_critical_authority_calls_resolve_without_unresolved_edges() -> None:
    report = analyze_strict_production_reachability()
    assert not report.unresolved_calls, "\n".join(
        f"{c.module_relpath}:{c.lineno} unresolved {c.resolved_target}" for c in report.unresolved_calls
    )


def test_p1_negative_fixture_detects_fake_adapter_instantiation() -> None:
    snippet = """
from intergrax.agent_distribution.deployment import FakeInMemoryRuntimeDeploymentAdapter

def evil():
    return FakeInMemoryRuntimeDeploymentAdapter()
"""
    violations = analyze_module_snippet(snippet)
    assert violations


def test_p1_negative_fixture_allows_isinstance_rejection_guard() -> None:
    snippet = """
from intergrax.agent_distribution.deployment import FakeInMemoryRuntimeDeploymentAdapter

def reject_reference_runtime_deployment_adapter(adapter):
    if isinstance(adapter, FakeInMemoryRuntimeDeploymentAdapter):
        raise ValueError("forbidden")
"""
    violations = analyze_module_snippet(snippet)
    assert not violations


def test_p1_negative_asgi_admission_docstring_only_does_not_pass() -> None:
    snippet = '''
def create_governed_contractor_strict_production_asgi_app(session):
    """session.readiness.ready_for_traffic must be checked."""
    from governed_contractor_application.host.main import create_governed_contractor_process_app
    return create_governed_contractor_process_app(process_composition=session.composition)
'''
    errors = analyze_asgi_admission_snippet(snippet)
    assert errors


def test_p1_negative_asgi_admission_calls_process_app_on_failure_path() -> None:
    snippet = '''
def create_governed_contractor_strict_production_asgi_app(session):
    if not session.readiness.ready_for_traffic:
        from governed_contractor_application.host.main import create_governed_contractor_process_app
        return create_governed_contractor_process_app(process_composition=session.composition)
    raise ValueError("unexpected")
'''
    errors = analyze_asgi_admission_snippet(snippet)
    assert any("failure path" in e for e in errors)


def test_p1_negative_asgi_admission_neutralized_guard_no_raise() -> None:
    snippet = '''
def create_governed_contractor_strict_production_asgi_app(session):
    if not session.readiness.ready_for_traffic:
        pass
    from governed_contractor_application.host.main import create_governed_contractor_process_app
    return create_governed_contractor_process_app(process_composition=session.composition)
'''
    errors = analyze_asgi_admission_snippet(snippet)
    assert errors


def test_p1_negative_asgi_admission_raise_only_in_else_branch() -> None:
    snippet = '''
def create_governed_contractor_strict_production_asgi_app(session):
    if not session.readiness.ready_for_traffic:
        pass
    else:
        raise ValueError("misplaced raise")
    from governed_contractor_application.host.main import create_governed_contractor_process_app
    return create_governed_contractor_process_app(process_composition=session.composition)
'''
    errors = analyze_asgi_admission_snippet(snippet)
    assert errors


def test_p1_negative_asgi_admission_raise_only_in_nested_optional_if() -> None:
    snippet = '''
def create_governed_contractor_strict_production_asgi_app(session):
    if not session.readiness.ready_for_traffic:
        if session.composition is not None:
            raise ValueError("optional nested guard")
    from governed_contractor_application.host.main import create_governed_contractor_process_app
    return create_governed_contractor_process_app(process_composition=session.composition)
'''
    errors = analyze_asgi_admission_snippet(snippet)
    assert errors


def test_p1_negative_asgi_admission_process_app_before_guard() -> None:
    snippet = '''
def create_governed_contractor_strict_production_asgi_app(session):
    from governed_contractor_application.host.main import create_governed_contractor_process_app
    app = create_governed_contractor_process_app(process_composition=session.composition)
    if not session.readiness.ready_for_traffic:
        raise ValueError("guard too late")
    return app
'''
    errors = analyze_asgi_admission_snippet(snippet)
    assert any("before" in e and "admission guard" in e for e in errors)


def test_p1_negative_gca_detects_direct_fake_adapter_constructor() -> None:
    snippet = """
from intergrax.agent_distribution.deployment import FakeInMemoryRuntimeDeploymentAdapter

def create_governed_contractor_strict_production_asgi_app(session):
    FakeInMemoryRuntimeDeploymentAdapter()
"""
    assert analyze_gca_deployment_snippet(snippet)


def test_p1_negative_gca_detects_aliased_fake_adapter_constructor() -> None:
    snippet = """
from intergrax.agent_distribution.deployment import (
    FakeInMemoryRuntimeDeploymentAdapter as AlternateAdapter,
)

def create_governed_contractor_strict_production_asgi_app(session):
    AlternateAdapter()
"""
    assert analyze_gca_deployment_snippet(snippet)


def test_p1_negative_gca_detects_qualified_fake_adapter_constructor() -> None:
    snippet = """
import intergrax.agent_distribution.deployment as deployment_mod

def create_governed_contractor_strict_production_asgi_app(session):
    deployment_mod.FakeInMemoryRuntimeDeploymentAdapter()
"""
    assert analyze_gca_deployment_snippet(snippet)


def test_p1_negative_gca_allows_isinstance_rejection_only() -> None:
    snippet = """
from intergrax.agent_distribution.deployment import FakeInMemoryRuntimeDeploymentAdapter

def reject_reference_runtime_deployment_adapter(adapter):
    if isinstance(adapter, FakeInMemoryRuntimeDeploymentAdapter):
        raise ValueError("forbidden")
"""
    assert not analyze_gca_deployment_snippet(snippet)


def test_p1_negative_fixture_detects_open_reference_lifecycle_call() -> None:
    snippet = """
from intergrax.applications._shared.reference_production_process_lifecycle import (
    open_reference_production_process_lifecycle,
)

def bypass():
    open_reference_production_process_lifecycle()
"""
    violations = analyze_module_snippet(snippet)
    assert violations


def test_p1_trusted_roots_documented() -> None:
    assert len(STRICT_PRODUCTION_ROOTS) >= 6
    roots = {r[1] for r in STRICT_PRODUCTION_ROOTS}
    assert "open_strict_production_serving_session" in roots
    assert "run" in roots
    assert PROFILE_V1_STRICT_STARTUP_OPENER in roots
    assert "run_package_b_strict_production_serving_command" in roots


def test_p1_negative_profile_v1_bypasses_admission_gate() -> None:
    snippet = """
def open_gca_strict_production_serving_session_from_single_node_production_profile_v1(
    environment, governance, deployment_adapter, **kwargs
):
    session = open_gca_strict_production_serving_session(
        storage=None,
        environment=environment,
        governance=governance,
        deployment_adapter=deployment_adapter,
    )
    session.bind_process_scoped_release_on_close(lambda: None)
    return session, None, None, None
"""
    errors = analyze_profile_v1_startup_snippet(snippet)
    assert any("load_validate_and_admit" in e for e in errors)


def test_p1_negative_profile_v1_alternative_strict_launcher() -> None:
    snippet = """
from intergrax.applications._shared.strict_production_serving_launcher import (
    open_strict_production_serving_session,
)
from intergrax.applications._shared.single_node_production_profile_v1 import (
    load_validate_and_admit_single_node_production_profile_v1_from_env,
)

def open_gca_strict_production_serving_session_from_single_node_production_profile_v1(
    environment, governance, deployment_adapter, **kwargs
):
    config, storage, writer_hold = (
        load_validate_and_admit_single_node_production_profile_v1_from_env()
    )
    try:
        session = open_strict_production_serving_session(
            storage=storage,
            environment=environment,
            governance=governance,
            deployment_adapter=deployment_adapter,
        )
    except BaseException:
        writer_hold.release()
        raise
    session.bind_process_scoped_release_on_close(writer_hold.release)
    return session, None, config, writer_hold
"""
    errors = analyze_profile_v1_startup_snippet(snippet)
    assert any("open_gca_strict_production_serving_session" in e for e in errors)


def test_p1_negative_profile_v1_reference_lifecycle_on_reachable_snippet() -> None:
    snippet = """
from intergrax.applications._shared.reference_production_process_lifecycle import (
    open_reference_production_process_lifecycle,
)

def open_gca_strict_production_serving_session_from_single_node_production_profile_v1():
    open_reference_production_process_lifecycle()
"""
    violations = analyze_module_snippet(snippet)
    assert violations


def test_p1_negative_profile_v1_unresolved_root_fails_locator_gate() -> None:
    """Declared root missing from boundary inventory must not pass root-locator gate."""
    import tests.qualification.prod_q._prod_q_r1_r7_q_p1_strict_reachability as p1_mod

    original = p1_mod.STRICT_PRODUCTION_ROOTS
    ghost = (
        (
            "applications/governed_contractor_application/host/single_node_profile_v1_strict_startup.py",
            "nonexistent_profile_v1_opener_for_qualification_only",
        ),
    )
    p1_mod.STRICT_PRODUCTION_ROOTS = original + ghost
    try:
        with pytest.raises(AssertionError, match="P1 roots unresolved"):
            assert_all_strict_production_roots_located()
    finally:
        p1_mod.STRICT_PRODUCTION_ROOTS = original


def test_p1_coverage_boundary_is_static_callsites_only() -> None:
    """Documented limitation: no dynamic dispatch or attribute call graph."""
    report = analyze_strict_production_reachability()
    assert report.reachable_functions, "expected non-empty bounded reachability set"
