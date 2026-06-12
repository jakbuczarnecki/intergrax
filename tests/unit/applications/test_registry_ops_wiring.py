# © Artur Czarnecki. All rights reserved.

"""APP-OPS-4 — application and environment registry wiring."""

from __future__ import annotations

from pathlib import Path

import pytest

from intergrax.applications._shared.product_manifest_registry import iter_product_manifests
from intergrax.applications._shared.registry_ops_wiring import (
    build_application_registry,
    build_environment_registry,
    get_application,
    list_environments,
    sync_platform_registries,
)
from intergrax.applications.contracts.application_package import ApplicationDistributionChannel
from intergrax.applications._shared.package_wiring import build_application_package, package_gate_environment

pytestmark = [pytest.mark.unit, pytest.mark.gate]

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_build_application_registry_covers_products() -> None:
    registry = build_application_registry(REPO_ROOT)
    expected = {manifest.app_id for _, manifest in iter_product_manifests()}
    assert {entry.app_id for entry in registry.entries} == expected
    for entry in registry.entries:
        assert entry.package_ref is not None
        assert entry.health is not None


def test_build_environment_registry_has_strict_hosts() -> None:
    registry = build_environment_registry(REPO_ROOT)
    assert registry.entries
    assert all(entry.environment_id.endswith("-strict") for entry in registry.entries)


def test_sync_and_query_helpers(tmp_path: Path) -> None:
    sync_platform_registries(REPO_ROOT)
    legal = get_application(REPO_ROOT, "legal")
    assert legal is not None
    envs = list_environments(REPO_ROOT, app_id="legal")
    assert any(item.environment_id == "legal-strict" for item in envs)


def test_register_application_upserts(tmp_path: Path) -> None:
    from intergrax.applications._shared.registry_ops_wiring import register_application

    _, manifest = next(iter(iter_product_manifests()))
    gate_env = package_gate_environment(manifest.resolved_environment())
    package = build_application_package(
        manifest,
        gate_env,
        channel=ApplicationDistributionChannel.LOCAL,
    )
    entry = register_application(tmp_path, package)
    assert entry.app_id == manifest.app_id
    assert (tmp_path / "build" / "application_registry.json").is_file()
