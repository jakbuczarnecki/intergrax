# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime


@dataclass(frozen=True)
class ChangeDossier:
    schema_version: str
    created_at_utc: str
    graph_digest: str
    inventory_digest: str
    stale: bool


def build_change_dossier(*, graph_digest: str, inventory_digest: str, stale: bool = False) -> ChangeDossier:
    return ChangeDossier(
        schema_version="evol_change_dossier.v1",
        created_at_utc=datetime.now(UTC).replace(microsecond=0).isoformat(),
        graph_digest=graph_digest,
        inventory_digest=inventory_digest,
        stale=stale,
    )
