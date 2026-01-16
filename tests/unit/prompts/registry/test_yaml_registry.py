# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from pathlib import Path

from intergrax.prompts.registry.yaml_registry import YamlPromptRegistry
from intergrax.prompts.registry.pin_config import PromptPinConfig
from intergrax.prompts.storage.models import PromptNotFound


def _make_prompt(p: Path, version: int) -> None:
    p.write_text(
        f"""
id: test
version: {version}

content:
  system: "s"
  user_template: "u"

meta:
  model_family: gpt-4
  output_schema_id: x
  tags: []
        """,
        encoding="utf-8",
    )


def test_registry_resolves_stable(tmp_path: Path) -> None:
    d = tmp_path / "test"
    d.mkdir()

    _make_prompt(d / "1.yaml", 1)
    _make_prompt(d / "2.yaml", 2)

    (d / "stable.yaml").write_text("stable: 2", encoding="utf-8")

    r = YamlPromptRegistry(tmp_path)
    r.load_all()

    lp = r.resolve("test")
    assert lp.document.version == 2


def test_registry_resolves_pinned(tmp_path: Path) -> None:
    d = tmp_path / "test"
    d.mkdir()

    _make_prompt(d / "1.yaml", 1)
    _make_prompt(d / "2.yaml", 2)

    (d / "stable.yaml").write_text("stable: 1", encoding="utf-8")

    r = YamlPromptRegistry(tmp_path)
    r.load_all()

    pin = PromptPinConfig(pins={"test": 2})

    lp = r.resolve("test", pin=pin)
    assert lp.document.version == 2


def test_missing_prompt(tmp_path: Path) -> None:
    r = YamlPromptRegistry(tmp_path)
    r.load_all()

    try:
        r.resolve("none")
    except PromptNotFound:
        pass
    else:
        raise AssertionError("Expected PromptNotFound")
