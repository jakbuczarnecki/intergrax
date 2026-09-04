"""Storage environment availability checks for real E2E qualification."""

from __future__ import annotations

import os


def postgres_environment_available() -> bool:
    dsn = os.getenv("INTERGRAX_POSTGRESQL_DSN", "").strip()
    host = os.getenv("INTERGRAX_POSTGRESQL_HOST", "").strip()
    return bool(dsn or host)


def qdrant_environment_available() -> bool:
    url = os.getenv("INTERGRAX_QDRANT_URL", "").strip()
    host = os.getenv("INTERGRAX_QDRANT_HOST", "").strip()
    return bool(url or host)


def storage_environment_available() -> bool:
    return postgres_environment_available() and qdrant_environment_available()


def storage_environment_gap() -> str | None:
    missing: list[str] = []
    if not postgres_environment_available():
        missing.append("INTERGRAX_POSTGRESQL_DSN or INTERGRAX_POSTGRESQL_HOST")
    if not qdrant_environment_available():
        missing.append("INTERGRAX_QDRANT_URL or INTERGRAX_QDRANT_HOST")
    if not missing:
        return None
    return "missing: " + ", ".join(missing)
