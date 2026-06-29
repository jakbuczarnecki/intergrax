# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.scaffold.application_names import ScaffoldApplicationNames, app_slug
from intergrax.scaffold.new_application import create_application

pytestmark = [pytest.mark.unit, pytest.mark.gate]


@pytest.mark.parametrize(
    ("raw", "pkg", "short", "env_prefix", "pascal"),
    [
        ("my_lab", "my_lab_application", "my_lab", "MY_LAB_", "MyLab"),
        ("MyLab", "mylab_application", "mylab", "MYLAB_", "Mylab"),
        ("my_lab_application", "my_lab_application", "my_lab", "MY_LAB_", "MyLab"),
        ("legal-poc", "legal_poc_application", "legal_poc", "LEGAL_POC_", "LegalPoc"),
    ],
)
def test_scaffold_name_normalization(
    raw: str,
    pkg: str,
    short: str,
    env_prefix: str,
    pascal: str,
) -> None:
    assert app_slug(raw) == pkg
    names = ScaffoldApplicationNames.resolve(raw, port=9001)
    assert names.pkg == pkg
    assert names.short == short
    assert names.env_prefix == env_prefix
    assert names.pascal == pascal
    assert names.tests_pkg == "tests"
    assert names.factory_fn == f"create_{short}_application"
    assert names.builders_const == f"{env_prefix.rstrip('_')}_AGENT_BUILDERS"


def test_scaffold_generated_code_uses_resolved_names(tmp_path) -> None:
    root = tmp_path / "repo"
    (root / "applications").mkdir(parents=True)
    names = ScaffoldApplicationNames.resolve(
        "Acme Demo",
        route_prefix="/v1/acme",
        port=7777,
    )
    target = create_application(
        name="Acme Demo",
        agents=["echo"],
        profile="lab",
        root=root,
        route_prefix="/v1/acme",
        port=7777,
        force=True,
    )

    assert target.name == names.pkg
    manifest = (target / "manifest.py").read_text(encoding="utf-8")
    assert f'app_id="{names.short}"' in manifest
    assert names.env_prefix in manifest
    assert (target / "host" / "settings.py").read_text(encoding="utf-8").count(
        f"class {names.settings_class}"
    ) == 1
    wiring = (target / "host" / "wiring.py").read_text(encoding="utf-8")
    assert names.builders_const in wiring
    assert f"def {names.registry_fn}" in wiring
    factory = (target / "host" / "factory.py").read_text(encoding="utf-8")
    assert f"def {names.factory_fn}" in factory
    env = (target / ".env.example").read_text(encoding="utf-8")
    assert f"{names.env_prefix}BACKEND_PORT=7777" in env
