# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Tier-3 application directory layout helpers (docs/, scripts/, sample_docs/)."""

from __future__ import annotations

from pathlib import Path

SAMPLE_DOCS_GITIGNORE = "*\n!.gitignore\n!.gitkeep\n"


def application_docs_dir(app_dir: Path) -> Path:
    return app_dir / "docs"


def write_sample_docs_scaffold(app_dir: Path, *, force: bool = True) -> None:
    """Create ``sample_docs/`` with gitignore rules for local smoke fixtures."""
    sample_dir = app_dir / "sample_docs"
    sample_dir.mkdir(parents=True, exist_ok=True)
    for name, content in (
        (".gitignore", SAMPLE_DOCS_GITIGNORE),
        (".gitkeep", ""),
    ):
        path = sample_dir / name
        if path.exists() and not force:
            raise FileExistsError(f"File already exists: {path}")
        path.write_text(content, encoding="utf-8")
