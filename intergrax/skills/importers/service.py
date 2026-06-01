# © Artur Czarnecki. All rights reserved.

"""Skill import service with optional trace recording."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from intergrax.runtime.events.context_skill_recording import record_skill_import_failed
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.skills.core.contracts import SkillManifest
from intergrax.skills.importers.cursor_skill_md import CursorSkillImportError, CursorSkillImporter


def import_cursor_skill_file(
    path: Path,
    *,
    event_bus: Optional[RuntimeEventBus] = None,
    default_skill_id: Optional[str] = None,
) -> SkillManifest:
    importer = CursorSkillImporter()
    try:
        return importer.import_file(path, default_skill_id=default_skill_id)
    except CursorSkillImportError as exc:
        if event_bus is not None:
            record_skill_import_failed(
                event_bus,
                source=str(path),
                reason=str(exc),
            )
        raise
