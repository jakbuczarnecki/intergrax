# © Artur Czarnecki. All rights reserved.

"""Typed Collaborative Work catalog materialization factory contract."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from intergrax.collaborative_work.persistence import CollaborativeWorkRepositories


@dataclass(frozen=True)
class CollaborativeWorkMaterializationBinding:
    """Typed materialization inputs resolved from Integrations profile options."""

    connection_factory: Callable[[], object] | None = None
    data_dir: Path | str | None = None
    relational_db: Path | str | None = None
    dsn: str | None = None
    host: str | None = None
    port: int | None = None
    user: str | None = None
    password: str | None = None
    database: str | None = None
    sslmode: str | None = None
    tenant_schema: str | None = None


@runtime_checkable
class CollaborativeWorkPersistenceFactory(Protocol):
    """Explicit catalog factory capability for Collaborative Work persistence."""

    def materialize_collaborative_work_repositories(
        self,
        binding: CollaborativeWorkMaterializationBinding,
    ) -> CollaborativeWorkRepositories:
        """Materialize the authoritative Collaborative Work repository bundle."""


def binding_from_profile_options(
    options: Mapping[str, Any],
) -> CollaborativeWorkMaterializationBinding:
    """Translate merged Integrations profile options into a typed CW binding."""
    connection_factory = options.get("connection_factory")
    data_dir = options.get("data_dir")
    relational_db = options.get("relational_db")
    return CollaborativeWorkMaterializationBinding(
        connection_factory=connection_factory if callable(connection_factory) else None,
        data_dir=data_dir,
        relational_db=relational_db,
        dsn=_optional_str(options.get("dsn")),
        host=_optional_str(options.get("host")),
        port=_optional_int(options.get("port")),
        user=_optional_str(options.get("user")),
        password=_optional_str(options.get("password")),
        database=_optional_str(options.get("database")),
        sslmode=_optional_str(options.get("sslmode")),
        tenant_schema=_optional_str(options.get("tenant_schema")),
    )


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return str(value)


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return int(value)
