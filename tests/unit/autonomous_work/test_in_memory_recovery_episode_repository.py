# © Artur Czarnecki. All rights reserved.

"""AW-6B — in-memory recovery episode repository contract tests."""

from __future__ import annotations

import pytest

from intergrax.autonomous_work.in_memory_recovery_episode_repository import (
    InMemoryWorkerRecoveryEpisodeRepository,
)
from tests.unit.autonomous_work import recovery_episode_repository_contracts as contracts

pytestmark = pytest.mark.unit


def _factory() -> InMemoryWorkerRecoveryEpisodeRepository:
    return InMemoryWorkerRecoveryEpisodeRepository()


def test_in_memory_recovery_episode_contracts() -> None:
    contracts.run_recovery_episode_repository_contract_suite(_factory)


def test_in_memory_recovery_episode_orphaned_claim() -> None:
    contracts.test_recovery_episode_orphaned_claim_blocks_next_attempt(_factory)


def test_in_memory_recovery_episode_concurrent_create() -> None:
    contracts.test_recovery_episode_concurrent_create_same_payload(_factory)


def test_in_memory_recovery_episode_terminal_restart() -> None:
    contracts.test_recovery_episode_terminal_restart(_factory)
