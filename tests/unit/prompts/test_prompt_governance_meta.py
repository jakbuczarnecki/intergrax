from __future__ import annotations

from pathlib import Path

import pytest

from intergrax.prompts.registry.governance_validation import validate_prompt_document_governance
from intergrax.prompts.registry.yaml_registry import YamlPromptRegistry
from intergrax.prompts.storage.yaml_loader import YamlPromptLoader


def test_yaml_loader_parses_governance_meta(tmp_path: Path) -> None:
    prompt_path = tmp_path / "1.yaml"
    prompt_path.write_text(
        """
id: harness.sample
version: 1
locales:
  en:
    content:
      system: "hello"
      developer: null
      user_template: null
meta:
  model_family: generic
  output_schema_id: harness.sample.v1
  tags: [harness]
  owner_team: platform
  owner_contact: harness@intergrax
  risk_tier: low
""",
        encoding="utf-8",
    )
    loaded = YamlPromptLoader().load(prompt_path)
    assert loaded.document.meta.owner_team == "platform"
    validation = validate_prompt_document_governance(loaded.document)
    assert validation.valid is True


def test_harness_reference_prompt_is_in_catalog() -> None:
    registry = YamlPromptRegistry.create_default(load=True)
    loaded = registry.resolve("harness_capability_summary")
    assert loaded.document.meta.owner_team == "platform"
    assert loaded.document.meta.risk_tier.value == "low"
