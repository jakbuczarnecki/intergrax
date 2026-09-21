# © Artur Czarnecki. All rights reserved.

"""OBS-DIAG-X5 / X5A — sqlite-file durable persistence restart + tenant isolation."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from intergrax.runtime.diagnostics.persistence_conformance import (
    assert_problem_persistence_conformance,
    assert_problem_persistence_typed_round_trip,
)
from testing_support.cross_process_spine.durable_document_store import (
    SqliteFileDocumentStore,
)
from testing_support.runtime.diagnostics.problem_persistence_test_support import (
    TEST_DOCUMENT_STORE_CURSOR_SECRET,
    document_store_problem_persistence_for_tests,
)

pytestmark = [pytest.mark.integration, pytest.mark.obs_diag_x5, pytest.mark.obs_diag_x5a]


def _sqlite_stack(db_path: Path) -> tuple[SqliteFileDocumentStore, object]:
    store = SqliteFileDocumentStore(
        db_path,
        cursor_secret=TEST_DOCUMENT_STORE_CURSOR_SECRET,
    )
    persistence = document_store_problem_persistence_for_tests(store)
    return store, persistence


def test_sqlite_file_problem_persistence_fresh_client_reads_durable_state(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "problems.sqlite3"
    writer_store, writer = _sqlite_stack(db_path)
    assert_problem_persistence_typed_round_trip(writer, label="x5-sqlite-file")
    writer_store.close()

    _reader_store, reader = _sqlite_stack(db_path)
    assert_problem_persistence_conformance(reader, label="x5-sqlite-file-reopen")


def test_sqlite_file_tenant_isolation_on_fresh_open(tmp_path: Path) -> None:
    db_path = tmp_path / "tenant.sqlite3"
    store, persistence = _sqlite_stack(db_path)
    assert_problem_persistence_conformance(persistence, label="x5-sqlite-tenant")
    store.close()


def test_sqlite_file_truncate_storage_removes_durable_state(tmp_path: Path) -> None:
    db_path = tmp_path / "truncate.sqlite3"
    store, persistence = _sqlite_stack(db_path)
    assert_problem_persistence_typed_round_trip(persistence, label="x5-sqlite-truncate")
    store.truncate_storage()
    with sqlite3.connect(db_path) as conn:
        row_count = conn.execute("SELECT COUNT(*) FROM diagnostic_documents").fetchone()[0]
    assert row_count == 0
    store.close()
