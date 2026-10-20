# © Artur Czarnecki. All rights reserved.

"""R7-Q §C dimension 11 — strict production graph reachability (closed-world).

Closed-world boundary: P1 strict production reachability modules plus projection/handoff
assembly modules wired from ``open_strict_production_serving_session`` and
``_bind_serving_handoff_for_environment``. Does not claim repo-wide call-graph completeness.

Reused evidence (admissible @ HEAD — not replayed via nested pytest):
- P1 static reachability: ``tests/qualification/prod_q/test_prod_q_r1_r7_q_p1_strict_production_reachability.py``
- D1b handoff negatives: ``tests/unit/applications/test_prod_q_r1_r7_d1b_serving_runtime_handoff.py``
- D3 descriptor authority: ``tests/unit/applications/test_prod_q_r1_r7_d3_*``
- DIM2 reference locator + digest: ``test_c_dim2_04`` in dim2/dim3 module
- DIM10 resolver identity: ``test_prod_q_r1_r7_q_c_dim10_asgi_runtime_identity_continuity_gate.py``

DIM16-DEPENDENCY (audit residual — not satisfied by DIM10 close tests):
``begin_drain()`` / ``begin_colocated_serving_execution_drain_on_strict_session_close``
failure may prevent strict session resource cleanup; DIM16 must inspect exception
handling, cleanup ordering, primary error preservation, and physical admission release.
"""

from __future__ import annotations

import ast
import subprocess
from pathlib import Path
from typing import Final

import pytest

from intergrax.agent_distribution.deployment import FakeInMemoryRuntimeDeploymentAdapter
from intergrax.applications._shared.production_registry_projection_input_bundle import (
    ProductionRegistryProjectionInputError,
    resolve_production_artifact_root,
)
from intergrax.applications._shared.strict_production_serving_launcher import (
    StrictProductionServingLauncherError,
    open_strict_production_serving_session,
)
from intergrax.applications._shared.venv_bundle_runtime_agent_factory_resolver import (
    VenvBundleRuntimeAgentFactoryResolver,
)
from tests.qualification.prod_q._prod_q_r1_r7_q_p1_strict_reachability import (
    BOUNDARY_MODULE_RELPATHS,
    analyze_strict_production_reachability,
    assert_startup_chain_opener,
    repo_root,
)
from tests.qualification.prod_q._prod_q_r1_r7_q_p2_strict_integrated_e2e import (
    load_d4_launcher_module,
)

pytestmark = [pytest.mark.qualification, pytest.mark.gate]

_EXPECTED_START_HEAD: Final = "42ffc6f7a40eda450a6a37351a2d8e39f6cd9b76"

_DIM16_LIFECYCLE_RESIDUAL: Final = (
    "DIM16-DEPENDENCY — drain failure may prevent strict session resource cleanup"
)

_DIM11_PROJECTION_WIRING_MODULES: Final[tuple[str, ...]] = (
    "intergrax/applications/_shared/strict_production_serving_launcher.py",
    "intergrax/applications/_shared/serving_runtime_handoff.py",
    "intergrax/applications/_shared/production_registry_projection_input_bundle.py",
    "intergrax/applications/_shared/production_host_composition.py",
    "intergrax/applications/_shared/deploy_activate_pipeline.py",
    "intergrax/applications/_shared/venv_bundle_runtime_agent_factory_resolver.py",
)

_STRICT_LAUNCHER_RELPATH: Final = (
    "intergrax/applications/_shared/strict_production_serving_launcher.py"
)

_FORBIDDEN_TEST_AUTHORITY_IMPORT_PREFIXES: Final[tuple[str, ...]] = (
    "testing_support.",
    "tests.",
)


def test_c_dim11_g00_qualification_baseline_ancestor() -> None:
    subprocess.check_call(
        ["git", "merge-base", "--is-ancestor", _EXPECTED_START_HEAD, "HEAD"],
        cwd=repo_root(),
    )


def test_c_dim11_g99_dim16_lifecycle_residual_recorded() -> None:
    """DIM16 scope marker — does not claim DIM16 PASS."""
    launcher = repo_root() / _STRICT_LAUNCHER_RELPATH
    text = launcher.read_text(encoding="utf-8")
    assert _DIM16_LIFECYCLE_RESIDUAL.split("—")[0].strip() in _DIM16_LIFECYCLE_RESIDUAL
    assert "begin_colocated_serving_execution_drain_on_strict_session_close" in text
    assert "close_process_resources" in text


def _parse_module(relpath: str) -> tuple[ast.Module, dict[str, str]]:
    path = repo_root() / relpath
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                local = alias.asname or alias.name
                imports[local] = f"{module}.{alias.name}" if module else alias.name
        elif isinstance(node, ast.Import):
            for alias in node.names:
                local = alias.asname or alias.name
                imports[local] = alias.name
    return tree, imports


def _function_def(tree: ast.Module, name: str) -> ast.FunctionDef | ast.AsyncFunctionDef:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name == name:
            return node
    raise AssertionError(f"missing function {name!r}")


def _call_targets(func: ast.AST, imports: dict[str, str]) -> set[str]:
    targets: set[str] = set()
    for node in ast.walk(func):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            bound = imports.get(node.func.id)
            if bound:
                targets.add(bound)
            targets.add(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            parts: list[str] = []
            cur: ast.expr = node.func
            while isinstance(cur, ast.Attribute):
                parts.append(cur.attr)
                cur = cur.value
            if isinstance(cur, ast.Name):
                base = imports.get(cur.id, cur.id)
                qual = ".".join([base, *reversed(parts)])
                targets.add(qual)
    return targets


def _imported_module_paths(tree: ast.Module) -> list[str]:
    modules: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module:
            modules.append(node.module)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                modules.append(alias.name)
    return modules


def test_c_dim11_01_production_graph_closure_reuses_p1_and_projection_wiring() -> None:
    """DIM11-01 — sanctioned opener chain; no prohibited harness/reference execution."""
    assert_startup_chain_opener()
    report = analyze_strict_production_reachability()
    assert not report.prohibited, "\n".join(
        f"{v.module_relpath}:{v.lineno} {v.qualname} -> {v.detail}" for v in report.prohibited
    )
    assert not report.gca_deployment_constructions
    assert not report.unresolved_calls

    expected_roots = {
        (
            "intergrax/applications/_shared/strict_production_serving_launcher.py",
            "open_strict_production_serving_session",
        ),
        (
            "intergrax/applications/_shared/production_process_lifecycle.py",
            "open_production_process_lifecycle",
        ),
        (
            "applications/governed_contractor_application/host/single_node_profile_v1_strict_startup.py",
            "open_gca_strict_production_serving_session_from_single_node_production_profile_v1",
        ),
        (
            "applications/governed_contractor_application/host/strict_gca_production_process_composition.py",
            "open_gca_strict_production_serving_session",
        ),
    }
    missing = expected_roots - report.reachable_functions
    assert not missing, f"missing reachable production roots: {missing}"

    tree, imports = _parse_module(_STRICT_LAUNCHER_RELPATH)
    opener = _function_def(tree, "open_strict_production_serving_session")
    opener_calls = _call_targets(opener, imports)
    assert any(c.endswith("open_production_process_lifecycle") for c in opener_calls)
    assert any(
        c.endswith("deploy_and_activate") or "deploy_and_activate" in c for c in opener_calls
    )

    bind_tree, bind_imports = _parse_module(_STRICT_LAUNCHER_RELPATH)
    bind_fn = _function_def(bind_tree, "_bind_serving_handoff_for_environment")
    bind_calls = _call_targets(bind_fn, bind_imports)
    assert any(c.endswith("resolve_production_artifact_root") for c in bind_calls)
    assert any(c.endswith("bind_colocated_serving_runtime_after_activation") for c in bind_calls)
    assert any("projection_input_store.get" in c or c.endswith(".get") for c in bind_calls)

    require_fn = _function_def(bind_tree, "_require_venv_production_adapter")
    require_calls = _call_targets(require_fn, bind_imports)
    assert any(c.endswith("reject_reference_runtime_deployment_adapter") for c in require_calls)


def test_c_dim11_02_no_reference_artifact_locator_at_strict_bind_boundary() -> None:
    """DIM11-02 — reference:// forbidden; strict bind path uses production resolver.

    Negative locator authority: reuse DIM2 ``test_c_dim2_04`` (same contract entrypoint).
    Additional strict-boundary assertion: handoff bind resolves materialization locators only
    through ``resolve_production_artifact_root`` (no parallel reference resolver).
    """
    with pytest.raises(ProductionRegistryProjectionInputError, match="reference://"):
        resolve_production_artifact_root("reference://artifacts/forbidden")

    tree, imports = _parse_module(_STRICT_LAUNCHER_RELPATH)
    bind_fn = _function_def(tree, "_bind_serving_handoff_for_environment")
    bind_calls = _call_targets(bind_fn, imports)
    assert any(c.endswith("resolve_production_artifact_root") for c in bind_calls)

    bundle_mod, _bundle_imports = _parse_module(
        "intergrax/applications/_shared/production_registry_projection_input_bundle.py",
    )
    resolve_fn = _function_def(bundle_mod, "resolve_production_artifact_root")
    resolve_src = ast.get_source_segment(
        (
            repo_root()
            / "intergrax/applications/_shared/production_registry_projection_input_bundle.py"
        ).read_text(encoding="utf-8"),
        resolve_fn,
    )
    assert resolve_src is not None
    assert 'startswith("reference://")' in resolve_src


def test_c_dim11_03_no_manual_registry_seed_authority_in_strict_boundary() -> None:
    """DIM11-03 — production boundary modules do not import test seed helpers."""
    modules = tuple(dict.fromkeys((*BOUNDARY_MODULE_RELPATHS, *_DIM11_PROJECTION_WIRING_MODULES)))
    violations: list[str] = []
    for relpath in modules:
        path = repo_root() / relpath
        assert path.is_file(), f"missing boundary module: {relpath}"
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(relpath))
        for imported in _imported_module_paths(tree):
            for prefix in _FORBIDDEN_TEST_AUTHORITY_IMPORT_PREFIXES:
                if imported.startswith(prefix):
                    violations.append(f"{relpath}: import {imported}")
        if "seed_canonical_serving_execution_stores" in path.read_text(encoding="utf-8"):
            violations.append(f"{relpath}: seed_canonical_serving_execution_stores")
    assert not violations, "\n".join(violations)

    host = repo_root() / "intergrax/applications/_shared/production_host_composition.py"
    host_text = host.read_text(encoding="utf-8")
    assert "bootstrap_production_registry_projection" in host_text
    assert "resolve_active_registry_projection" in host_text
    assert "HarnessHostRegistryAuthorityError" in host_text


def test_c_dim11_04_factory_resolver_from_validated_handoff_not_injection(
    tmp_path: Path,
) -> None:
    """DIM11-04 — handoff resolver is VenvBundle production resolver aligned to unit."""
    d4, err = load_d4_launcher_module()
    assert err is None, f"integration import blocked: {err}"
    session = d4._open_with_activation(
        tmp_path,
        environment=d4._ac3_environment(),
        revision_id="rev-c-dim11-resolver",
    )
    try:
        handoff = session.serving_runtime_handoff
        resolver = handoff.runtime_agent_factory_resolver
        assert isinstance(resolver, VenvBundleRuntimeAgentFactoryResolver)
        revision_id = "rev-c-dim11-resolver"
        stores = session.composition.agent_platform_runtime.stores
        materialization = stores.materialization_store.get_by_revision(revision_id)
        assert materialization is not None
        assert resolver.artifact_root == resolve_production_artifact_root(
            materialization.artifact_locator,
        )
        projection_input = session.lifecycle.lifecycle_services.projection_input_store.get(
            revision_id,
        )
        assert projection_input is not None
        assert handoff.runtime_agent_factory_resolver is resolver

        tree, imports = _parse_module(_STRICT_LAUNCHER_RELPATH)
        bind_fn = _function_def(tree, "_bind_serving_handoff_for_environment")
        bind_src = ast.get_source_segment(
            (repo_root() / _STRICT_LAUNCHER_RELPATH).read_text(encoding="utf-8"),
            bind_fn,
        )
        assert bind_src is not None
        assert "runtime_agent_factory_resolver_for_unit" in bind_src
        assert "factory_resolver=unit_resolver" in bind_src
    finally:
        session.close()


def test_c_dim11_05_fail_closed_without_authoritative_projection_or_reference_fallback(
    tmp_path: Path,
) -> None:
    """DIM11-05 — missing serving authority / fake adapter; no reference runtime fallback."""
    d4, err = load_d4_launcher_module()
    assert err is None, f"integration import blocked: {err}"
    environment = d4._ac3_environment()
    storage, adapter = d4._storage_and_adapter(tmp_path)
    from intergrax.applications.contracts.environment_profile import (
        ApplicationEnvironmentProfile,
    )

    governance = d4._governance(
        ApplicationEnvironmentProfile(profile_id=environment.application_environment_id),
    )

    with pytest.raises(StrictProductionServingLauncherError):
        open_strict_production_serving_session(
            storage=storage,
            environment=environment,
            governance=governance,
            deployment_adapter=adapter,
            composition_builder=d4._composition_builder(tmp_path),
            missing_continuation_file_policy="empty",
        )

    with pytest.raises(
        StrictProductionServingLauncherError,
        match="FakeInMemoryRuntimeDeploymentAdapter",
    ):
        open_strict_production_serving_session(
            storage=storage,
            environment=environment,
            governance=governance,
            deployment_adapter=FakeInMemoryRuntimeDeploymentAdapter(),
        )

    report = analyze_strict_production_reachability()
    ref_calls = [v for v in report.prohibited if "open_reference" in v.detail]
    assert not ref_calls
