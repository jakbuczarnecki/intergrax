# © Artur Czarnecki. All rights reserved.

"""Import Cursor-style SKILL.md files into ``SkillManifest`` (Phase R-Skill.8)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from intergrax.skills.core.contracts import SkillManifest, SkillRiskTier

_FRONT_MATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


class CursorSkillImportError(ValueError):
    """Raised when SKILL.md cannot be parsed into a valid manifest."""


class CursorSkillImporter:
    """
    Best-effort importer for Cursor ``SKILL.md`` packs.

    Expected frontmatter keys (all optional except implied skill_id from path):
    - ``name`` or ``skill_id``
    - ``description``
    - ``version``
    - ``tools`` (comma-separated or YAML list as comma-separated line)
    - ``risk_tier``
  """

    def import_file(self, path: Path, *, default_skill_id: str | None = None) -> SkillManifest:
        text = path.read_text(encoding="utf-8")
        return self.import_text(text, default_skill_id=default_skill_id or self._skill_id_from_path(path))

    def import_text(self, text: str, *, default_skill_id: str) -> SkillManifest:
        frontmatter, _body = self._split_frontmatter(text)
        meta = self._parse_frontmatter(frontmatter)
        skill_id = str(meta.get("skill_id") or meta.get("name") or default_skill_id).strip()
        if not skill_id:
            raise CursorSkillImportError("skill_id is required")
        description = str(meta.get("description", "")).strip()
        if not description:
            raise CursorSkillImportError("description is required in SKILL.md frontmatter")
        version = str(meta.get("version", "1.0.0")).strip()
        tool_ids = self._parse_tool_ids(meta.get("tools"))
        risk_raw = str(meta.get("risk_tier", "low")).strip().lower()
        try:
            risk_tier = SkillRiskTier(risk_raw)
        except ValueError as exc:
            raise CursorSkillImportError(f"invalid risk_tier: {risk_raw}") from exc
        return SkillManifest(
            skill_id=skill_id,
            version=version,
            description=description,
            tool_ids=tool_ids,
            risk_tier=risk_tier,
        )

    @staticmethod
    def _skill_id_from_path(path: Path) -> str:
        return path.parent.name.replace("_", ".")

    @staticmethod
    def _split_frontmatter(text: str) -> tuple[str, str]:
        match = _FRONT_MATTER_RE.match(text)
        if not match:
            raise CursorSkillImportError("SKILL.md must start with YAML frontmatter (---)")
        return match.group(1), text[match.end() :]

    @staticmethod
    def _parse_frontmatter(block: str) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for line in block.splitlines():
            if ":" not in line:
                continue
            key, _, value = line.partition(":")
            result[key.strip()] = value.strip().strip('"').strip("'")
        return result

    @staticmethod
    def _parse_tool_ids(raw: object) -> tuple[str, ...]:
        if raw is None:
            return ()
        if isinstance(raw, list):
            return tuple(str(item).strip() for item in raw if str(item).strip())
        text = str(raw).strip()
        if not text:
            return ()
        if text.startswith("[") and text.endswith("]"):
            inner = text[1:-1]
            return tuple(part.strip().strip("'\"") for part in inner.split(",") if part.strip())
        return tuple(part.strip() for part in text.split(",") if part.strip())
