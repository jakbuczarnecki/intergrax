# © Artur Czarnecki. All rights reserved.

"""TRACE-X-P5-R2-P4-R1-R1-R1-R1-R1-R1 pin ambiguous-outcome contract tests (P4 API)."""

from __future__ import annotations

import inspect
from typing import Any

import pytest

from intergrax.applications._shared.integrations.persistence import (
    DocumentStoreExecutionIntegrationConfigurationPinningStore,
    InMemoryExecutionIntegrationConfigurationPinningStore,
    KvExecutionIntegrationConfigurationPinningStore,
)
pytestmark = pytest.mark.unit

_SKIP = "TRACE-X-P5-R2-P4 extended pin API (read_pin_records + staging) not implemented"


def _extended_pin_api_available() -> bool:
    store = InMemoryExecutionIntegrationConfigurationPinningStore()
    return callable(getattr(store, "read_pin_records", None)) and "requirement_recovery_staging" in inspect.signature(
        store.pin,
    ).parameters


@pytest.fixture
def pin_stores() -> list[Any]:
    pytest.importorskip("intergrax.applications._shared.integrations.persistence")
    from tests.unit.applications.integrations.test_trace_x_p5_r2_p2_persistence import (
        InMemoryConditionalDocumentStore,
        InMemoryKVStore,
    )

    return [
        InMemoryExecutionIntegrationConfigurationPinningStore(),
        KvExecutionIntegrationConfigurationPinningStore(InMemoryKVStore()),
        DocumentStoreExecutionIntegrationConfigurationPinningStore(InMemoryConditionalDocumentStore()),
    ]


@pytest.mark.skipif(not _extended_pin_api_available(), reason=_SKIP)
def test_ambiguous_pin_lost_acknowledgement_retry_reuses_stored_staging() -> None:
    """§8 — durable pin accepted, caller retries with fresh local candidate → reconcile to S1."""
    raise NotImplementedError("P4 pin recovery staging implementation")


@pytest.mark.skipif(not _extended_pin_api_available(), reason=_SKIP)
def test_ambiguous_pin_no_row_retry_may_create_new_staging() -> None:
    """§9 — no durable row; new staging candidate allowed on first successful pin."""
    raise NotImplementedError("P4 pin recovery staging implementation")


@pytest.mark.skipif(not _extended_pin_api_available(), reason=_SKIP)
def test_ambiguous_pin_concurrent_same_obligation_loser_reconciles_winner_staging(
    pin_stores: list[Any],
) -> None:
    """§10 — CAS winner; loser adopts stored staging after read."""
    del pin_stores
    raise NotImplementedError("P4 pin recovery staging implementation")


@pytest.mark.skipif(not _extended_pin_api_available(), reason=_SKIP)
def test_ambiguous_pin_concurrent_different_provenance_conflict() -> None:
    """§10 — semantic provenance mismatch → CONFLICT."""
    raise NotImplementedError("P4 pin recovery staging implementation")


@pytest.mark.skipif(not _extended_pin_api_available(), reason=_SKIP)
def test_ambiguous_pin_stored_staging_never_overwritten() -> None:
    """§14.5 — pin with divergent staging on existing row → CONFLICT."""
    raise NotImplementedError("P4 pin recovery staging implementation")


@pytest.mark.skipif(not _extended_pin_api_available(), reason=_SKIP)
def test_ambiguous_pin_restart_recovers_staging_before_spine() -> None:
    """§14.6 — read_pin_records after restart before spine append."""
    raise NotImplementedError("P4 pin recovery staging implementation")


def test_ambiguous_pin_no_second_durable_staging_store_module() -> None:
    """§14.7 — no second durable staging store."""
    from pathlib import Path

    persistence = (
        Path(__file__).resolve().parents[4]
        / "intergrax/applications/_shared/integrations/persistence.py"
    )
    source = persistence.read_text(encoding="utf-8")
    assert "RequirementRecoveryStagingStore" not in source
    assert "recovery_staging_store" not in source.lower()


@pytest.mark.docker
@pytest.mark.skipif(not _extended_pin_api_available(), reason=_SKIP)
def test_ambiguous_pin_docker_backed_write_recovery() -> None:
    """§14.8 — Docker-backed ambiguous-write recovery (production persistence path)."""
    raise NotImplementedError("P4 pin recovery staging implementation")
