# © Artur Czarnecki. All rights reserved.

"""``new-integration`` scaffold must emit INTEGRATIONS-2E provider layout."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from intergrax.scaffold.integration_templates import class_prefix
from intergrax.scaffold.new_integration import run_new_integration

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.no_ci]


def test_new_integration_scaffold_emits_runtime_contract_layout(tmp_path: Path) -> None:
    root = tmp_path
    args = argparse.Namespace(
        slug="acme_kv",
        category="key_value_cache",
        root=root,
        force=False,
    )
    assert run_new_integration(args) == 0

    provider_dir = root / "intergrax" / "integrations" / "providers" / "key_value_cache" / "acme_kv"
    expected_files = {"integration.py", "bundle.py", "register.py", "manifest.py", "USAGE.md"}
    assert expected_files == {path.name for path in provider_dir.iterdir()}

    legacy_files = {"adapter.py", "plugin.py"}
    assert legacy_files.isdisjoint(path.name for path in provider_dir.iterdir())

    integration = (provider_dir / "integration.py").read_text(encoding="utf-8")
    assert "class AcmeKvKeyValueCacheIntegration(" in integration
    assert "IntegrationPlugin" not in integration
    assert "AcmeKvKeyValueCacheClient" in integration

    bundle = (provider_dir / "bundle.py").read_text(encoding="utf-8")
    assert "def create_acme_kv_key_value_cache_integration(" in bundle
    assert "def create_acme_kv_key_value_cache(" in bundle
    assert "IntegrationPlugin" not in bundle

    register = (provider_dir / "register.py").read_text(encoding="utf-8")
    assert "register_from_manifest" in register
    assert "register_integration_plugin" not in register


def test_new_integration_scaffold_observability_backend(tmp_path: Path) -> None:
    root = tmp_path
    args = argparse.Namespace(
        slug="acme_obs",
        category="observability_backend",
        root=root,
        force=False,
    )
    assert run_new_integration(args) == 0

    provider_dir = root / "intergrax" / "integrations" / "providers" / "observability_backend" / "acme_obs"
    integration = (provider_dir / "integration.py").read_text(encoding="utf-8")
    assert class_prefix("acme_obs", "observability_backend") + "Integration" in integration
    assert "ObservabilityVendorIntegrationContract" in integration
    assert "from_transport" in integration

    bundle = (provider_dir / "bundle.py").read_text(encoding="utf-8")
    assert "create_acme_obs_observability_integration" in bundle
    assert "create_acme_obs_observability_backend" in bundle


def test_new_integration_rejects_unknown_category(tmp_path: Path) -> None:
    args = argparse.Namespace(
        slug="bad",
        category="not_a_category",
        root=tmp_path,
        force=False,
    )
    assert run_new_integration(args) == 1
