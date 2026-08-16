# © Artur Czarnecki. All rights reserved.

"""PLATFORM-PLUGIN-5: configuration, secrets and DI convention contract tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from intergrax.core.plugins.errors import PlatformPluginManifestValidationError
from intergrax.core.plugins.manifest_io import parse_platform_plugin_manifest_data

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_ARCH_PATH = _REPO_ROOT / "docs" / "project" / "architecture" / "PLATFORM_PLUGINS.md"
_AUTHOR_GUIDE_PATH = (
    _REPO_ROOT / "docs" / "project" / "technical" / "guides" / "EXTENSION_AUTHOR_GUIDE.md"
)
_ROADMAP_PATH = _REPO_ROOT / "docs" / "project" / "maintainers" / "plans" / "PLATFORM_PLUGINS.md"


def test_architecture_contains_config_di_matrix() -> None:
    text = _ARCH_PATH.read_text(encoding="utf-8")
    assert "### 12.3 Cross-surface configuration, secrets and DI matrix" in text
    assert "### 12.4 Canonical configuration flow" in text
    for surface in (
        "IntegrationPlugin",
        "ToolPlugin",
        "SkillPlugin",
        "ContextPlugin",
        "SecurityDefensePlugin",
        "PolicyRuleHandler",
        "ToolInvocationPattern",
        "VendorKnowledge",
    ):
        assert surface in text
    assert "ToolWiringContext" in text
    assert "env_prefix" in text


def test_architecture_rejects_global_di_and_secret_api() -> None:
    text = _ARCH_PATH.read_text(encoding="utf-8")
    assert "global Platform Plugin DI container" in text
    assert "get_secret()" in text
    assert "service locator" in text.lower()
    matrix_start = text.index("### 12.3 Cross-surface configuration")
    matrix_section = text[matrix_start : text.index("## 13. Dependency injection", matrix_start)]
    assert "**Rejected in PLUGIN-5:**" in matrix_section


def test_author_guide_config_credentials_di_section() -> None:
    text = _AUTHOR_GUIDE_PATH.read_text(encoding="utf-8")
    assert "## 14. Configuration, credentials and dependency injection" in text
    assert "[tool.intergrax.plugin]" in text
    assert "ToolWiringContext" in text
    assert "env_prefix" in text
    assert "side-effect free" in text


def test_roadmap_marks_plugin_5_done() -> None:
    text = _ROADMAP_PATH.read_text(encoding="utf-8")
    assert "**PLATFORM-PLUGIN-5**" in text
    assert "**Done**" in text.split("PLATFORM-PLUGIN-5")[1].split("PLATFORM-PLUGIN-6")[0]


def test_nested_runtime_credentials_rejected_in_manifest() -> None:
    with pytest.raises(PlatformPluginManifestValidationError, match="secret-like manifest field"):
        parse_platform_plugin_manifest_data({"options": {"api_key": "x"}})
