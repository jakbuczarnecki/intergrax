# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.observability_backend.langfuse import (
    LANGFUSE_OBSERVABILITY_PROVIDER_ID,
    LangfuseObservabilityIntegration,
    LangfuseObservabilityIntegrationConfig,
    LangfuseObservabilityTransport,
    create_langfuse_observability_backend,
    create_langfuse_observability_integration,
    register_langfuse_integration,
)
from intergrax.integrations.providers.observability_backend.langfuse.bundle import (
    create_langfuse_observability_integration as bundle_create_integration,
)
from intergrax.integrations.providers.observability_backend.langfuse.register import (
    register_langfuse_integration as register_fn,
)
from scripts.maintenance._provider_shell_contract import (
    HAND_EDITED_PROVIDER_FILES,
    is_contract_aware_package,
    should_skip_provider_file,
)
from scripts.maintenance.wire_p3_provider_shells import generate_provider_shell

pytestmark = pytest.mark.unit

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
_LANGFUSE_PKG = (
    _PROJECT_ROOT
    / "intergrax"
    / "integrations"
    / "providers"
    / "observability_backend"
    / "langfuse"
)
_CANONICAL_LAYOUT = (
    "integration.py",
    "manifest.py",
    "bundle.py",
    "register.py",
    "__init__.py",
    "USAGE.md",
)
_FORBIDDEN_VENDOR_IMPORT_PREFIXES = ("langfuse",)


def test_langfuse_package_follows_canonical_provider_layout() -> None:
    for filename in _CANONICAL_LAYOUT:
        assert (_LANGFUSE_PKG / filename).is_file(), f"missing {filename}"


def test_langfuse_integration_module_contains_contract_class() -> None:
    source = (_LANGFUSE_PKG / "integration.py").read_text(encoding="utf-8")
    assert "class LangfuseObservabilityIntegration" in source


def test_langfuse_bundle_exposes_legacy_and_contract_factories() -> None:
    bundle_source = (_LANGFUSE_PKG / "bundle.py").read_text(encoding="utf-8")
    assert "create_langfuse_observability_backend" in bundle_source
    assert "create_langfuse_observability_integration" in bundle_source
    assert "__all__" in bundle_source
    assert callable(create_langfuse_observability_backend)
    assert callable(bundle_create_integration)


def test_langfuse_register_uses_legacy_factory_only() -> None:
    source = inspect.getsource(register_fn)
    assert "create_langfuse_observability_backend" in source
    assert "create_langfuse_observability_integration" not in source


def test_langfuse_create_integration_enabled_without_transport_fails_early() -> None:
    with pytest.raises(IntegrationConfigurationError, match="transport"):
        create_langfuse_observability_integration(enabled=True, transport=None)


def test_langfuse_create_integration_disabled_without_transport_allowed() -> None:
    integration = create_langfuse_observability_integration(enabled=False, transport=None)
    assert integration.config.enabled is False
    assert integration.transport is None


def test_wire_p3_does_not_overwrite_langfuse_integration_py() -> None:
    integration_path = _LANGFUSE_PKG / "integration.py"
    before = integration_path.read_text(encoding="utf-8")
    generate_provider_shell(
        "langfuse",
        "OBSERVABILITY_BACKEND",
        factory="create_langfuse_observability_backend",
        env="INTERGRAX_LANGFUSE",
        providers_root=_PROJECT_ROOT / "intergrax" / "integrations" / "providers",
    )
    after = integration_path.read_text(encoding="utf-8")
    assert before == after


def test_wire_p3_preserves_langfuse_contract_factory_exports() -> None:
    bundle_path = _LANGFUSE_PKG / "bundle.py"
    before = bundle_path.read_text(encoding="utf-8")
    generate_provider_shell(
        "langfuse",
        "OBSERVABILITY_BACKEND",
        factory="create_langfuse_observability_backend",
        env="INTERGRAX_LANGFUSE",
        providers_root=_PROJECT_ROOT / "intergrax" / "integrations" / "providers",
    )
    after = bundle_path.read_text(encoding="utf-8")
    assert "create_langfuse_observability_integration" in after
    assert before == after


def test_langfuse_init_lazy_exports_public_api() -> None:
    assert LANGFUSE_OBSERVABILITY_PROVIDER_ID == "langfuse"
    assert LangfuseObservabilityIntegration.__name__ == "LangfuseObservabilityIntegration"
    assert LangfuseObservabilityIntegrationConfig.__name__ == "LangfuseObservabilityIntegrationConfig"
    assert LangfuseObservabilityTransport.__name__ == "LangfuseObservabilityTransport"
    assert callable(create_langfuse_observability_backend)
    assert callable(create_langfuse_observability_integration)
    assert callable(register_langfuse_integration)


def test_langfuse_integration_module_has_no_vendor_sdk_imports() -> None:
    source = (_LANGFUSE_PKG / "integration.py").read_text(encoding="utf-8").lower()
    for token in _FORBIDDEN_VENDOR_IMPORT_PREFIXES:
        assert f"import {token}" not in source
        assert f"from {token}" not in source


def test_contract_aware_package_skips_hand_edited_files() -> None:
    assert is_contract_aware_package(_LANGFUSE_PKG)
    for filename in HAND_EDITED_PROVIDER_FILES:
        assert should_skip_provider_file(_LANGFUSE_PKG, filename)
