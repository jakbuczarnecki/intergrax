# © Artur Czarnecki. All rights reserved.

"""Scaffold product profile — canonical registry lifecycle enforcement."""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from intergrax.scaffold.new_application import create_application
from tests.unit.applications.ac3_projection_helpers import build_test_registry_projection
from tests.unit.applications.scaffold_runtime_helper import (
    factory_callable,
    import_scaffold_modules,
    prepare_scaffold_package,
    product_settings_class,
)

pytestmark = [pytest.mark.unit, pytest.mark.agent_os, pytest.mark.gate]

_FORBIDDEN_PRODUCT_PATTERNS = (
    "build_application_registry(",
    "AgentRegistry(",
    "registry.register(",
    "from_agents(",
    'model_copy(update={"contract_id": settings.default_agent_id})',
    "APPLICATION_MANIFEST.agents[0]",
)


def _assert_no_forbidden_patterns(target: Path) -> None:
    host_dir = target / "host"
    for path in host_dir.rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        for pattern in _FORBIDDEN_PRODUCT_PATTERNS:
            assert pattern not in source, f"{path.relative_to(target)} contains forbidden {pattern!r}"


def test_product_scaffold_has_no_registry_construction_bypass(tmp_path) -> None:
    target = create_application(
        name="lifecycle_product",
        agents=["echo"],
        profile="product",
        root=tmp_path,
        port=8201,
        force=True,
    )
    _assert_no_forbidden_patterns(target)
    assert not (target / "host" / "wiring.py").exists()

    factory_src = (target / "host" / "factory.py").read_text(encoding="utf-8")
    assert "registry_projection: MaterializedRegistryProjection" in factory_src
    assert "runtime.registry" in factory_src


def test_product_scaffold_preserves_binding_identity_without_rewrite(tmp_path) -> None:
    target = create_application(
        name="lifecycle_identity",
        agents=["echo", "signoff_probe"],
        profile="product",
        root=tmp_path,
        port=8202,
        force=True,
    )
    manifest = (target / "manifest.py").read_text(encoding="utf-8")
    assert "AgentBinding.mount(EchoAgent" in manifest
    assert "AgentBinding.mount(SignoffProbeAgent" in manifest
    assert 'model_copy(update={"contract_id"' not in manifest
    _assert_no_forbidden_patterns(target)


@pytest.mark.no_ci
def test_product_scaffold_projection_resolves_revision_bound_registry(tmp_path) -> None:
    import importlib

    from intergrax.applications._shared.harness_registry_authority import (
        RegistryAssemblyMode,
        resolve_harness_host_registry,
    )
    from intergrax.applications.contracts.build_context import ApplicationBuildContext
    from intergrax.scaffold.application_names import ScaffoldApplicationNames

    _, pkg, short = prepare_scaffold_package(
        tmp_path,
        name="lifecycle_runtime",
        profile="product",
        port=8203,
        route_prefix="/v1/lifecycle_runtime",
        agents=["echo", "signoff_probe"],
    )
    factory_mod, settings_mod = import_scaffold_modules(pkg)
    create_backend = factory_callable(factory_mod, short, product=True)
    signature = inspect.signature(create_backend)
    assert signature.parameters["registry_projection"].default is inspect.Parameter.empty

    manifest_mod = importlib.import_module(f"{pkg}.manifest")
    env_mod = importlib.import_module(f"{pkg}.host.environment_profile")
    builders_mod = importlib.import_module(f"{pkg}.host.agent_builders")
    manifest = manifest_mod.__dict__[f"build_{short}_manifest"]()
    settings = product_settings_class(settings_mod, short).from_env()
    env = manifest.environment or env_mod.__dict__[f"build_{short}_environment_profile"](settings)
    names = ScaffoldApplicationNames.resolve(short, port=8203)
    projection = build_test_registry_projection(
        manifest,
        env,
        builders=builders_mod.__dict__[names.builders_const],
    )
    registry, evidence = resolve_harness_host_registry(
        manifest=manifest,
        build_context=ApplicationBuildContext.for_manifest(manifest),
        environment=env,
        assembly_mode=RegistryAssemblyMode.REVISION_BOUND,
        registry_projection=projection,
    )
    assert evidence is not None
    assert len(registry.list_agent_ids()) >= 2


@pytest.mark.no_ci
def test_product_scaffold_fail_closed_without_projection(tmp_path) -> None:
    _, pkg, short = prepare_scaffold_package(
        tmp_path,
        name="lifecycle_fail_closed",
        profile="product",
        port=8204,
        route_prefix="/v1/lifecycle_fail_closed",
    )
    factory_mod, _ = import_scaffold_modules(pkg)
    create_backend = factory_callable(factory_mod, short, product=True)
    with pytest.raises(TypeError):
        create_backend()  # type: ignore[call-arg]


def test_lab_scaffold_uses_shared_development_registry_bootstrap(tmp_path) -> None:
    target = create_application(
        name="lifecycle_lab",
        agents=["echo", "signoff_probe"],
        profile="lab",
        root=tmp_path,
        port=8205,
        force=True,
    )
    wiring = (target / "host" / "wiring.py").read_text(encoding="utf-8")
    assert "build_manifest_development_registry" in wiring
    assert "build_application_registry" not in wiring
    manifest = (target / "manifest.py").read_text(encoding="utf-8")
    assert "AgentBinding.mount(EchoAgent" in manifest
    assert "AgentBinding.mount(SignoffProbeAgent" in manifest
