# © Artur Czarnecki. All rights reserved.

"""Shared GR-6 host tests — seeded in-memory Collaborative Work repository injection."""

from __future__ import annotations

from datetime import UTC, datetime

from governed_contractor_application.host.collaborative_work_local_fixture import (
    build_in_memory_collaborative_work_repositories,
    seed_external_work_collaborative_governance_state,
)
from intergrax.collaborative_work.persistence import CollaborativeWorkRepositories

_DEFAULT_FIXTURE_CLOCK = datetime(2026, 6, 15, 12, 0, tzinfo=UTC)


def gr6_seeded_collaborative_work_repositories(
    *,
    tenant_id: str,
    workspace_id: str,
    principal_id: str,
) -> CollaborativeWorkRepositories:
    """Injected in-memory authoritative state for GR-6 production composition tests."""
    repositories = build_in_memory_collaborative_work_repositories()
    seed_external_work_collaborative_governance_state(
        repositories,
        tenant_id=tenant_id,
        workspace_id=workspace_id,
        principal_id=principal_id,
    )
    return repositories


def gr6_fixture_authority_clock() -> datetime:
    return _DEFAULT_FIXTURE_CLOCK
