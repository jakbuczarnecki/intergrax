# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Tier-2 agent directory layout helpers (docs/, docs/journal/)."""

from __future__ import annotations

from pathlib import Path


def agent_docs_dir(agent_dir: Path) -> Path:
    return agent_dir / "docs"


def write_agent_journal_scaffold(agent_dir: Path, *, force: bool = True) -> None:
    """Create ``docs/journal/`` for agent-local implementation history."""
    journal_dir = agent_docs_dir(agent_dir) / "journal"
    journal_dir.mkdir(parents=True, exist_ok=True)
    gitkeep = journal_dir / ".gitkeep"
    if gitkeep.exists() and not force:
        raise FileExistsError(f"File already exists: {gitkeep}")
    gitkeep.write_text("", encoding="utf-8")
