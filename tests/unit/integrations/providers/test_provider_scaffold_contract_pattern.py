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
from scripts.maintenance.wire_p3_provider_shells import generate_provider_shell as generate_p3_provider_shell
from scripts.maintenance.wire_p5_m6_p4_providers import generate_provider_shell as generate_p5_provider_shell
from scripts.maintenance.wire_p6_m6_p5_providers import generate_provider_shell as generate_p6_provider_shell
from scripts.maintenance.wire_p7_m6_p6_providers import generate_provider_shell as generate_p7_provider_shell

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
_LANGFUSE_SHELL_KWARGS = {
    "slug": "langfuse",
    "cat_enum": "OBSERVABILITY_BACKEND",
    "factory": "create_langfuse_observability_backend",
    "env": "INTERGRAX_LANGFUSE",
}
_GRAFANA_SHELL_KWARGS = {
    "slug": "grafana",
    "cat_enum": "OBSERVABILITY_BACKEND",
    "factory": "create_grafana_observability_backend",
    "env": "INTERGRAX_GRAFANA",
}
_OTEL_COLLECTOR_SHELL_KWARGS = {
    "slug": "opentelemetry_collector",
    "category": "observability_backend",
    "cat_enum": "OBSERVABILITY_BACKEND",
    "factory": "create_opentelemetry_collector_observability_backend",
    "env": "INTERGRAX_OPENTELEMETRY_COLLECTOR",
}
_NEWRELIC_SHELL_KWARGS = {
    "slug": "newrelic",
    "category": "observability_backend",
    "cat_enum": "OBSERVABILITY_BACKEND",
    "factory": "create_newrelic_observability_backend",
    "env": "INTERGRAX_NEWRELIC",
}
_P5_CANONICAL_FILES = ("manifest.py", "register.py", "bundle.py", "__init__.py", "USAGE.md")


def _contract_aware_langfuse_pkg(tmp_path: Path) -> Path:
    """Isolated contract-aware package mirroring migrated provider layout."""
    return _contract_aware_observability_pkg(
        tmp_path,
        "langfuse",
        factory="create_langfuse_observability_backend",
        integration_factory="create_langfuse_observability_integration",
    )


def _contract_aware_observability_pkg(
    tmp_path: Path,
    slug: str,
    *,
    factory: str,
    integration_factory: str,
) -> Path:
    """Isolated contract-aware package for P5/P6/P7 scaffold tests."""
    pkg = tmp_path / "providers" / "observability_backend" / slug
    pkg.mkdir(parents=True)
    (pkg / "integration.py").write_text(
        "class ExampleObservabilityIntegration:\n    pass\n",
        encoding="utf-8",
    )
    (pkg / "bundle.py").write_text(
        f"def {factory}(): ...\n"
        f"def {integration_factory}(): ...\n",
        encoding="utf-8",
    )
    (pkg / "manifest.py").write_text("MANIFEST = None\n", encoding="utf-8")
    (pkg / "register.py").write_text(f"def register_{slug}_integration(): pass\n", encoding="utf-8")
    (pkg / "__init__.py").write_text("__all__ = []\n", encoding="utf-8")
    (pkg / "USAGE.md").write_text("# usage\n", encoding="utf-8")
    return pkg


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


def test_wire_p3_does_not_overwrite_contract_aware_integration_py(tmp_path: Path) -> None:
    pkg = _contract_aware_langfuse_pkg(tmp_path)
    integration_path = pkg / "integration.py"
    before = integration_path.read_text(encoding="utf-8")

    written = generate_p3_provider_shell(
        **_LANGFUSE_SHELL_KWARGS,
        providers_root=tmp_path / "providers",
    )

    assert integration_path.read_text(encoding="utf-8") == before
    assert written["register.py"] is False
    assert written["bundle.py"] is False
    assert written["__init__.py"] is False


def test_wire_p3_preserves_contract_aware_bundle_exports(tmp_path: Path) -> None:
    pkg = _contract_aware_langfuse_pkg(tmp_path)
    bundle_path = pkg / "bundle.py"
    before = bundle_path.read_text(encoding="utf-8")

    generate_p3_provider_shell(
        **_LANGFUSE_SHELL_KWARGS,
        providers_root=tmp_path / "providers",
    )

    after = bundle_path.read_text(encoding="utf-8")
    assert "create_langfuse_observability_integration" in after
    assert before == after


def test_wire_p3_writes_legacy_shell_for_unmigrated_provider(tmp_path: Path) -> None:
    providers_root = tmp_path / "providers"

    written = generate_p3_provider_shell(
        **_LANGFUSE_SHELL_KWARGS,
        providers_root=providers_root,
    )

    pkg = providers_root / "observability_backend" / "langfuse"
    assert written == {"register.py": True, "bundle.py": True, "__init__.py": True}
    assert (pkg / "register.py").is_file()
    assert (pkg / "bundle.py").is_file()
    assert (pkg / "__init__.py").is_file()
    assert "create_langfuse_observability_backend" in (pkg / "bundle.py").read_text(encoding="utf-8")


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


def test_contract_aware_package_skips_hand_edited_files(tmp_path: Path) -> None:
    pkg = _contract_aware_langfuse_pkg(tmp_path)
    assert is_contract_aware_package(pkg)
    for filename in HAND_EDITED_PROVIDER_FILES:
        assert should_skip_provider_file(pkg, filename)


def test_wire_p5_preserves_contract_aware_package_files(tmp_path: Path) -> None:
    pkg = _contract_aware_observability_pkg(
        tmp_path,
        "grafana",
        factory="create_grafana_observability_backend",
        integration_factory="create_grafana_observability_integration",
    )
    before = {name: (pkg / name).read_text(encoding="utf-8") for name in _P5_CANONICAL_FILES}

    written = generate_p5_provider_shell(
        **_GRAFANA_SHELL_KWARGS,
        providers_root=tmp_path / "providers",
    )

    for name in _P5_CANONICAL_FILES:
        assert (pkg / name).read_text(encoding="utf-8") == before[name]
        assert written[name] is False


def test_wire_p6_preserves_contract_aware_package_files(tmp_path: Path) -> None:
    pkg = _contract_aware_observability_pkg(
        tmp_path,
        "opentelemetry_collector",
        factory="create_opentelemetry_collector_observability_backend",
        integration_factory="create_opentelemetry_collector_observability_integration",
    )
    before = {name: (pkg / name).read_text(encoding="utf-8") for name in _P5_CANONICAL_FILES}

    written = generate_p6_provider_shell(
        **_OTEL_COLLECTOR_SHELL_KWARGS,
        providers_root=tmp_path / "providers",
    )

    for name in _P5_CANONICAL_FILES:
        assert (pkg / name).read_text(encoding="utf-8") == before[name]
        assert written[name] is False


def test_wire_p7_preserves_contract_aware_package_files(tmp_path: Path) -> None:
    pkg = _contract_aware_observability_pkg(
        tmp_path,
        "newrelic",
        factory="create_newrelic_observability_backend",
        integration_factory="create_newrelic_observability_integration",
    )
    before = {name: (pkg / name).read_text(encoding="utf-8") for name in _P5_CANONICAL_FILES}

    written = generate_p7_provider_shell(
        **_NEWRELIC_SHELL_KWARGS,
        providers_root=tmp_path / "providers",
    )

    for name in _P5_CANONICAL_FILES:
        assert (pkg / name).read_text(encoding="utf-8") == before[name]
        assert written[name] is False


def test_wire_p5_writes_legacy_shell_for_unmigrated_provider(tmp_path: Path) -> None:
    providers_root = tmp_path / "providers"

    written = generate_p5_provider_shell(
        **_GRAFANA_SHELL_KWARGS,
        providers_root=providers_root,
    )

    pkg = providers_root / "observability_backend" / "grafana"
    assert written == dict.fromkeys(_P5_CANONICAL_FILES, True)
    for name in _P5_CANONICAL_FILES:
        assert (pkg / name).is_file()
    assert "create_grafana_observability_backend" in (pkg / "bundle.py").read_text(encoding="utf-8")


def test_wire_p6_writes_legacy_shell_for_unmigrated_provider(tmp_path: Path) -> None:
    providers_root = tmp_path / "providers"

    written = generate_p6_provider_shell(
        **_OTEL_COLLECTOR_SHELL_KWARGS,
        providers_root=providers_root,
    )

    pkg = providers_root / "observability_backend" / "opentelemetry_collector"
    assert written == dict.fromkeys(_P5_CANONICAL_FILES, True)
    for name in _P5_CANONICAL_FILES:
        assert (pkg / name).is_file()
    assert "create_opentelemetry_collector_observability_backend" in (pkg / "bundle.py").read_text(
        encoding="utf-8",
    )


def test_wire_p7_writes_legacy_shell_for_unmigrated_provider(tmp_path: Path) -> None:
    providers_root = tmp_path / "providers"

    written = generate_p7_provider_shell(
        **_NEWRELIC_SHELL_KWARGS,
        providers_root=providers_root,
    )

    pkg = providers_root / "observability_backend" / "newrelic"
    assert written == dict.fromkeys(_P5_CANONICAL_FILES, True)
    for name in _P5_CANONICAL_FILES:
        assert (pkg / name).is_file()
    assert "create_newrelic_observability_backend" in (pkg / "bundle.py").read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("generate_fn", "kwargs"),
    [
        (
            generate_p5_provider_shell,
            {
                "slug": "grafana",
                "cat_enum": "OBSERVABILITY_BACKEND",
                "factory": "create_grafana_observability_backend",
                "integration_factory": "create_grafana_observability_integration",
                "shell_kwargs": _GRAFANA_SHELL_KWARGS,
            },
        ),
        (
            generate_p6_provider_shell,
            {
                "slug": "opentelemetry_collector",
                "factory": "create_opentelemetry_collector_observability_backend",
                "integration_factory": "create_opentelemetry_collector_observability_integration",
                "shell_kwargs": _OTEL_COLLECTOR_SHELL_KWARGS,
            },
        ),
        (
            generate_p7_provider_shell,
            {
                "slug": "newrelic",
                "factory": "create_newrelic_observability_backend",
                "integration_factory": "create_newrelic_observability_integration",
                "shell_kwargs": _NEWRELIC_SHELL_KWARGS,
            },
        ),
    ],
)
def test_p5_p6_p7_preserves_contract_based_bundle_factory(
    tmp_path: Path,
    generate_fn,
    kwargs: dict[str, object],
) -> None:
    slug = str(kwargs["slug"])
    factory = str(kwargs["factory"])
    integration_factory = str(kwargs["integration_factory"])
    shell_kwargs = dict(kwargs["shell_kwargs"])  # type: ignore[arg-type]
    pkg = _contract_aware_observability_pkg(
        tmp_path,
        slug,
        factory=factory,
        integration_factory=integration_factory,
    )
    bundle_path = pkg / "bundle.py"
    before = bundle_path.read_text(encoding="utf-8")

    generate_fn(**shell_kwargs, providers_root=tmp_path / "providers")

    after = bundle_path.read_text(encoding="utf-8")
    assert integration_factory in after
    assert before == after
