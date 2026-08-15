# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

import re
from pathlib import Path

import pytest

from intergrax.scaffold.application_configuration_doc import (
    PLATFORM_CONFIGURATION_RELATIVE,
    render_application_configuration_doc,
)
from intergrax.scaffold.application_names import ScaffoldApplicationNames
from intergrax.scaffold.application_setting_specs import application_env_names, application_setting_specs
from intergrax.scaffold.new_application import create_application

pytestmark = [pytest.mark.unit, pytest.mark.agent_os, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SETTING_HEADING = re.compile(r"^### ([A-Z][A-Z0-9_]*)$", re.MULTILINE)
_ENV_ASSIGNMENT = re.compile(r"^(?:#\s*)?([A-Z][A-Z0-9_]*)=", re.MULTILINE)
_PLATFORM_ENV_CALL = re.compile(
    r'env\.(?:optional_str|str|int|bool|raw)\(\s*"([A-Z0-9_]+)"'
)
_FOREIGN_MARKERS = (
    "POC_TEMPLATE_",
    "LEGAL_",
    "LAB_APPLICATION",
    "CONCEPT_LAB_",
    "YOUR_APP",
    "FIXME",
    "<application>",
)


def _documented_env_names(text: str) -> tuple[str, ...]:
    return tuple(_SETTING_HEADING.findall(text))


def _env_example_prefixed_names(text: str, env_prefix: str) -> set[str]:
    return {name for name in _ENV_ASSIGNMENT.findall(text) if name.startswith(env_prefix)}


def _assert_configuration_contract(
    target: Path,
    *,
    names: ScaffoldApplicationNames,
    profile: str,
) -> None:
    config_path = target / "docs" / "CONFIGURATION.md"
    assert config_path.is_file()
    doc = config_path.read_text(encoding="utf-8")
    env_example = (target / ".env.example").read_text(encoding="utf-8")
    settings_src = (target / "host" / "settings.py").read_text(encoding="utf-8")
    readme = (target / "README.md").read_text(encoding="utf-8")

    expected = application_env_names(profile, names.env_prefix)
    documented = _documented_env_names(doc)

    assert f"# {names.display} Configuration" in doc
    assert names.pkg in doc
    assert names.env_prefix in doc
    assert documented == expected
    assert "docs/CONFIGURATION.md" in readme
    assert PLATFORM_CONFIGURATION_RELATIVE in doc
    assert "INTERGRAX_LLM_PROVIDER" in doc
    assert "INTERGRAX_LLM_MODEL" in doc
    assert "### INTERGRAX_LLM_PROVIDER" not in doc
    assert "### INTERGRAX_LLM_MODEL" not in doc
    assert "### INTERGRAX_EMBEDDING_PROVIDER" not in doc
    assert "### INTERGRAX_EMBEDDING_MODEL" not in doc
    assert all(name in _env_example_prefixed_names(env_example, names.env_prefix) for name in expected)
    assert "ApplicationSettingSpec" in doc
    assert "intergrax/scaffold/application_setting_specs.py" in doc
    assert "generated from the shared spec" in doc
    assert "not generated from `ApplicationSettingSpec`" in doc
    assert "Document the new variable in this file" not in doc
    assert "Add the same variable to `.env.example`" not in doc
    assert f'env_prefix: ClassVar[str] = "{names.env_prefix}"' in settings_src
    assert f'route_prefix: str = "{names.route_prefix}"' in settings_src
    assert f"backend_port: int = {names.port}" in settings_src

    resolved = (config_path.parent / PLATFORM_CONFIGURATION_RELATIVE).resolve()
    expected_platform = (
        _REPO_ROOT / "docs" / "project" / "technical" / "guides" / "PLATFORM_CONFIGURATION.md"
    ).resolve()
    layout_resolved = (
        _REPO_ROOT / "applications" / names.pkg / "docs" / PLATFORM_CONFIGURATION_RELATIVE
    ).resolve()
    assert layout_resolved == expected_platform
    assert expected_platform.is_file()
    assert resolved.name == "PLATFORM_CONFIGURATION.md"

    for marker in _FOREIGN_MARKERS:
        assert marker not in doc
        assert marker not in env_example

    if profile == "product":
        for spec in application_setting_specs("product"):
            if spec.env_suffix in {item.env_suffix for item in application_setting_specs("lab")}:
                continue
            assert f'"{spec.env_suffix}"' in settings_src
        assert f"{names.pascal}BackendSettings" in doc
        assert "DEFAULT_AGENT_ID" in doc
    else:
        assert f"{names.pascal}ApplicationSettings" in doc
        assert f"{names.env_prefix}DEFAULT_AGENT_ID" not in documented


def test_shared_setting_specs_match_settings_base_source() -> None:
    src = (
        _REPO_ROOT / "intergrax" / "applications" / "contracts" / "settings.py"
    ).read_text(encoding="utf-8")
    platform_fn = src.split("def _load_platform_env", 1)[1].split("def _load_app_env", 1)[0]
    loaded = _PLATFORM_ENV_CALL.findall(platform_fn)
    spec_suffixes = [spec.env_suffix for spec in application_setting_specs("lab")]
    assert spec_suffixes == list(dict.fromkeys(spec_suffixes))
    assert set(spec_suffixes) == set(loaded)
    assert len(spec_suffixes) == len(loaded)


def test_lab_scaffold_emits_application_configuration(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    (root / "applications").mkdir(parents=True)
    names = ScaffoldApplicationNames.resolve("config3_lab", route_prefix="/v1/config3_lab", port=8093)
    target = create_application(
        name="config3_lab",
        agents=["echo"],
        profile="lab",
        root=root,
        port=8093,
        route_prefix="/v1/config3_lab",
    )
    _assert_configuration_contract(target, names=names, profile="lab")


def test_product_scaffold_emits_application_configuration(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    (root / "applications").mkdir(parents=True)
    names = ScaffoldApplicationNames.resolve(
        "config3_product",
        route_prefix="/v1/config3_product",
        port=8002,
    )
    target = create_application(
        name="config3_product",
        agents=["echo"],
        profile="product",
        root=root,
        port=8002,
        route_prefix="/v1/config3_product",
    )
    _assert_configuration_contract(target, names=names, profile="product")


def test_minimal_lab_scaffold_emits_configuration_doc(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    (root / "applications").mkdir(parents=True)
    target = create_application(
        name="config3_minimal",
        agents=["echo"],
        profile="lab",
        root=root,
        force=True,
        minimal=True,
    )
    assert (target / "docs" / "CONFIGURATION.md").is_file()
    doc = (target / "docs" / "CONFIGURATION.md").read_text(encoding="utf-8")
    assert "Config3 Minimal Configuration" in doc
    assert "CONFIG3_MINIMAL_" in doc


def test_configuration_renderer_uses_caller_names_not_placeholders() -> None:
    names = ScaffoldApplicationNames.resolve("alpha_probe", route_prefix="/v1/alpha", port=8111)
    doc = render_application_configuration_doc(names=names, profile="lab")
    assert "Alpha Probe Configuration" in doc
    assert "ALPHA_PROBE_BACKEND_PORT=8111" in doc
    assert "ALPHA_PROBE_ROUTE_PREFIX=/v1/alpha" in doc
    assert "poc_template" not in doc.lower()
    assert "### INTERGRAX_LLM_PROVIDER" not in doc


def test_configuration_guidance_points_to_setting_specs_not_manual_duplication() -> None:
    lab_names = ScaffoldApplicationNames.resolve("alpha_probe", route_prefix="/v1/alpha", port=8111)
    product_names = ScaffoldApplicationNames.resolve(
        "alpha_product", route_prefix="/v1/alpha_product", port=8002
    )
    lab_doc = render_application_configuration_doc(names=lab_names, profile="lab")
    product_doc = render_application_configuration_doc(names=product_names, profile="product")
    for doc, settings_class in (
        (lab_doc, "AlphaProbeApplicationSettings"),
        (product_doc, "AlphaProductBackendSettings"),
    ):
        assert "ApplicationSettingSpec" in doc
        assert "intergrax/scaffold/application_setting_specs.py" in doc
        assert "generated from the shared spec" in doc
        assert "not generated from `ApplicationSettingSpec`" in doc
        assert settings_class in doc
        assert "Document the new variable in this file" not in doc
        assert "Add the same variable to `.env.example`" not in doc
        assert "Add a typed field on" not in doc
