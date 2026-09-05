# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.scaffold.new_application import create_application

pytestmark = [pytest.mark.unit, pytest.mark.agent_os, pytest.mark.gate]


def test_scaffold_product_profile_creates_fastapi_core_tree(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "applications").mkdir()

    target = create_application(
        name="demo_product",
        agents=["echo"],
        profile="product",
        root=root,
        port=8001,
        route_prefix="/v1/demo",
    )

    pkg = "demo_product_application"
    assert (target / "host" / "agent_factories.py").is_file()
    assert (target / "serving" / "schemas.py").is_file()

    manifest = (target / "manifest.py").read_text(encoding="utf-8")
    assert "ApplicationManifest.product" in manifest
    assert "default=True" in manifest
    assert "build_demo_product_echo_from_context" in manifest

    agent_factories = (target / "host" / "agent_factories.py").read_text(
        encoding="utf-8"
    )
    assert "ApplicationBuildContext" in agent_factories
    assert "AgentBinding" in agent_factories
    assert "def build_demo_product_echo_from_context(" in agent_factories

    factory = (target / "host" / "factory.py").read_text(encoding="utf-8")
    assert "create_demo_product_backend_app" in factory
    assert "create_app" in factory
    assert "create_debug_app" not in factory

    main_py = (target / "host" / "main.py").read_text(encoding="utf-8")
    assert "create_demo_product_process_app" in main_py
    assert "ProductionProcessComposition" in main_py
    assert "build_production_agent_platform_runtime" not in main_py

    settings = (target / "host" / "settings.py").read_text(encoding="utf-8")
    assert "DemoProductBackendSettings" in settings
    assert "BACKEND_BOOTSTRAP_API_KEY" in settings

    dockerfile = (target / "docker" / "Dockerfile").read_text(encoding="utf-8")
    assert f"{pkg}.host.main:app" in dockerfile

    deploy_doc = (target / "docs" / "BUILD_AND_DEPLOY.md").read_text(encoding="utf-8")
    assert "/health" in deploy_doc

    smoke = (target / "tests" / "host" / "test_demo_product_host_smoke.py").read_text(
        encoding="utf-8"
    )
    assert "test_demo_product_backend_requires_registry_projection_parameter" in smoke
    assert "create_demo_product_backend_app" in smoke
    assert "TestClient" not in smoke

    assert not (target / "host" / "wiring.py").exists()
    for path in (target / "host").rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        assert "build_application_registry(" not in source
        assert "AgentRegistry(" not in source
        assert (
            'model_copy(update={"contract_id": settings.default_agent_id})'
            not in source
        )

    env_example = (target / ".env.example").read_text(encoding="utf-8")
    assert "DEMO_PRODUCT_BACKEND_ENV=dev" in env_example
    assert "DEMO_PRODUCT_DEFAULT_AGENT_ID=echo" in env_example

    env_profile = (target / "host" / "environment_profile.py").read_text(
        encoding="utf-8"
    )
    assert "ApiEnvironment.DEV" in env_profile
    assert "middleware_hook_timeout_seconds" in env_profile
    assert "use_in_memory_trace=trace_db_path is None" in factory
