# © Artur Czarnecki. All rights reserved.

"""PROD-Q-R1-R3 — GCA strict startup evidence vs SQLite environment isolation."""

from __future__ import annotations

from pathlib import Path

import pytest

from governed_contractor_application.host.factory import create_governed_contractor_backend_app
from tests.qualification.prod_q._p0_certification import (
    build_r1_sqlite_disposition,
    classify_sqlite_header_bytes,
    probe_default_relational_db_header,
    probe_sqlite_header_at_path,
    r1_http_reproduction_requested,
)
from tests.qualification.prod_q._p0_types import (
    ProductionReadinessEvidenceState,
    SqliteReproductionClassification,
)
from governed_contractor_application.tests.governed_contractor_ac3_projection import (
    build_governed_contractor_test_registry_projection,
)
from intergrax.contracts.execution_continuation import (
    ExecutionContinuationError,
    ExecutionContinuationErrorCode,
)
from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.runtime.execution.continuation.persistence import (
    ExecutionContinuationDurableBacking,
    execution_continuation_state_store_from_durable_export,
    export_durable_continuation_state,
)
from testing_support.host_fixture_wiring import install_diagnostic_cursor_secret

pytestmark = [pytest.mark.unit]

def _durable_continuation_store():
    return execution_continuation_state_store_from_durable_export(
        export_durable_continuation_state(ExecutionContinuationDurableBacking()),
    )


def test_corrupt_sqlite_fixture_identified_under_tmp_path(tmp_path: Path) -> None:
    corrupt = tmp_path / "corrupt-relational.db"
    corrupt.write_bytes(b"%3|not-sqlite|FAIL|artifact")
    assert probe_sqlite_header_at_path(corrupt) != "SQLITE_HEADER"
    assert classify_sqlite_header_bytes(corrupt.read_bytes()[:16]) != "SQLITE_HEADER"


def test_default_db_probe_is_diagnostic_only_and_never_fails_qualification() -> None:
    build_r1_sqlite_disposition.cache_clear()
    _ = probe_default_relational_db_header()
    record = build_r1_sqlite_disposition()
    assert record.evidence_state == ProductionReadinessEvidenceState.BLOCKER
    assert record.default_db_header_probe in {
        "MISSING",
        "UNREADABLE",
        "SQLITE_HEADER",
    } or len(record.default_db_header_probe) >= 2


def test_missing_default_db_probe_reports_missing_without_failure(tmp_path: Path) -> None:
    missing = tmp_path / "no-default.db"
    assert probe_sqlite_header_at_path(missing) == "MISSING"


def test_valid_sqlite_fixture_reports_sqlite_header(tmp_path: Path) -> None:
    import sqlite3

    valid = tmp_path / "valid-relational.db"
    with sqlite3.connect(valid) as connection:
        connection.execute("CREATE TABLE prod_q_valid (id INTEGER PRIMARY KEY)")
        connection.commit()
    assert probe_sqlite_header_at_path(valid) == "SQLITE_HEADER"


def test_fast_disposition_skips_http_reproduction_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("INTERGRAX_PROD_Q_R1_HTTP_REPRODUCTION", raising=False)
    build_r1_sqlite_disposition.cache_clear()
    assert r1_http_reproduction_requested() is False
    record = build_r1_sqlite_disposition()
    assert record.reproduction_executed is False
    assert record.reproduction_classification == SqliteReproductionClassification.NOT_EXECUTED
    assert record.evidence_state == ProductionReadinessEvidenceState.BLOCKER


def test_isolated_relational_db_missing_continuation_fails_before_sqlite_bootstrap(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Scenario A — isolation env does not repair harness; continuation gate runs first."""
    install_diagnostic_cursor_secret(monkeypatch)
    isolated = tmp_path / "isolated-relational.db"
    monkeypatch.setenv("INTERGRAX_RELATIONAL_DB", str(isolated))
    with pytest.raises(ExecutionContinuationError) as exc:
        create_governed_contractor_backend_app(
            registry_projection=build_governed_contractor_test_registry_projection(),
            document_store=InMemoryDocumentStore(),
        )
    assert exc.value.code is ExecutionContinuationErrorCode.NON_DURABLE_CONTINUATION_STORE
    assert not isolated.exists()


def test_intergrax_relational_db_env_isolates_relational_path_from_polluted_default(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Scenario A — env injection selects a fresh relational file independent of build/intergrax.db."""
    import sqlite3

    from intergrax.integrations.providers.relational_store.sqlite.config import (
        SQLiteIntegrationConfig,
    )
    from intergrax.integrations.providers.relational_store.sqlite.paths import (
        resolve_sqlite_store_paths,
    )

    isolated = tmp_path / "isolated-relational.db"
    monkeypatch.setenv("INTERGRAX_RELATIONAL_DB", str(isolated))
    paths = resolve_sqlite_store_paths(
        SQLiteIntegrationConfig(data_dir=tmp_path / "data_dir"),
    )
    assert paths.relational == isolated
    with sqlite3.connect(paths.relational) as connection:
        connection.execute("CREATE TABLE IF NOT EXISTS prod_q_r1_r3_probe (id INTEGER PRIMARY KEY)")
        connection.commit()
    assert paths.relational.read_bytes()[:16].startswith(b"SQLite format 3")


def test_strict_lightweight_path_accepts_durable_continuation_before_harness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Scenario C — sanctioned factory wiring with explicit durable continuation (harness mocked)."""
    from unittest.mock import patch

    install_diagnostic_cursor_secret(monkeypatch)
    captured: dict[str, object] = {}

    def _capture_harness_build(*_args, **kwargs):
        captured.update(kwargs)
        raise RuntimeError("stop-after-harness-boundary")

    with patch(
        "governed_contractor_application.host.factory.build_harness_host_runtime",
        side_effect=_capture_harness_build,
    ):
        with pytest.raises(RuntimeError, match="stop-after-harness-boundary"):
            create_governed_contractor_backend_app(
                registry_projection=build_governed_contractor_test_registry_projection(),
                document_store=InMemoryDocumentStore(),
                execution_continuation_state_store=_durable_continuation_store(),
            )
    assert captured.get("execution_continuation_state_store") is not None
