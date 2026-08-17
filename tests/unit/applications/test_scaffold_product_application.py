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

    smoke = (
        target / "tests" / "host" / "test_demo_product_host_smoke.py"
    ).read_text(encoding="utf-8")
    assert "test_demo_product_backend_health" in smoke
    assert 'create_demo_product_backend_app' in smoke

    env_example = (target / ".env.example").read_text(encoding="utf-8")
    assert "DEMO_PRODUCT_BACKEND_ENV=dev" in env_example
    assert "DEMO_PRODUCT_DEFAULT_AGENT_ID=echo" in env_example

    env_profile = (target / "host" / "environment_profile.py").read_text(encoding="utf-8")
    assert "ApiEnvironment.DEV" in env_profile
    assert "middleware_hook_timeout_seconds" in env_profile
    assert "use_in_memory_trace=trace_db_path is None" in factory
