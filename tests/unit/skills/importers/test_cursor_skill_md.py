# © Artur Czarnecki. All rights reserved.

import pytest

from intergrax.skills.importers.cursor_skill_md import CursorSkillImportError, CursorSkillImporter


@pytest.mark.unit
def test_cursor_skill_importer_parses_frontmatter() -> None:
    text = """---
name: demo.skill
description: Demo skill pack
version: 2.0.0
tools: rag.retrieve, websearch.query
risk_tier: medium
---
# Body
"""
    manifest = CursorSkillImporter().import_text(text, default_skill_id="fallback")
    assert manifest.skill_id == "demo.skill"
    assert manifest.version == "2.0.0"
    assert manifest.tool_ids == ("rag.retrieve", "websearch.query")


@pytest.mark.unit
def test_cursor_skill_importer_requires_frontmatter() -> None:
    with pytest.raises(CursorSkillImportError):
        CursorSkillImporter().import_text("# no frontmatter", default_skill_id="x")
