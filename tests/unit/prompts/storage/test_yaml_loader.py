# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from pathlib import Path
import pytest

from intergrax.prompts.storage.yaml_loader import YamlPromptLoader
from intergrax.prompts.storage.models import PromptValidationError


def test_loader_parses_minimal_document(tmp_path: Path) -> None:
    p = tmp_path / "p.yaml"

    p.write_text("""
id: test
version: 1
locales:
  en:
    content:
      system: "hello"
      developer: null
      user_template: "u"
meta:
  model_family: "gpt-4"
  output_schema_id: "x"
  tags: ["a","b"]
""")


    doc = YamlPromptLoader().load(p)

    assert doc.document.id == "test"
    assert doc.document.version == 1
    assert doc.document.locales['en'].system == "hello"
    assert "a" in doc.document.meta.tags


def test_missing_fields_fail(tmp_path: Path) -> None:
    p = tmp_path / "bad.yaml"
    p.write_text("{}", encoding="utf-8")

    with pytest.raises(PromptValidationError):
        YamlPromptLoader().load(p)
