# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""SQLite integration configuration (Phase M.4)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from intergrax.integrations._shared.config import BaseIntegrationConfig

ENV_SQLITE_DATA_DIR = "INTERGRAX_SQLITE_DATA_DIR"

DEFAULT_DATA_DIR = Path("build")


class SQLiteIntegrationConfig(BaseIntegrationConfig):
    """
    SQLite layout for lab / local Tier-3 deployments.

    ``data_dir`` holds default ``*.db`` filenames; per-store paths may override
    via fields below or legacy ``INTERGRAX_*_DB`` env vars (see ``paths.py``).
    """

    data_dir: Path = DEFAULT_DATA_DIR
    relational_db: Optional[Path] = None
    trace_db: Optional[Path] = None
    runtime_events_db: Optional[Path] = None
    task_checkpoints_db: Optional[Path] = None
    human_decisions_db: Optional[Path] = None
    task_memory_db: Optional[Path] = None
    experiments_db: Optional[Path] = None
    idempotency_db: Optional[Path] = None
    session_db: Optional[Path] = None
    organization_db: Optional[Path] = None

    @classmethod
    def from_env(cls, **overrides: object) -> SQLiteIntegrationConfig:
        data_dir_raw = os.environ.get(ENV_SQLITE_DATA_DIR, "").strip()
        payload: dict[str, object] = {}
        if data_dir_raw:
            payload["data_dir"] = Path(data_dir_raw)
        payload.update(overrides)
        return cls.model_validate(payload)
