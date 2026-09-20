# © Artur Czarnecki. All rights reserved.

"""OBS-DIAG-X5 — sqlite-file durable persistence restart + tenant isolation."""

from __future__ import annotations

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

pytestmark = [pytest.mark.integration, pytest.mark.obs_diag_x5]


def _sqlite_problem_persistence(db_path: Path):
    store = SqliteFileDocumentStore(
        db_path,
        cursor_secret=TEST_DOCUMENT_STORE_CURSOR_SECRET,
    )
    return document_store_problem_persistence_for_tests(store)


def test_sqlite_file_problem_persistence_fresh_client_reads_durable_state(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "problems.sqlite3"
    writer = _sqlite_problem_persistence(db_path)
    assert_problem_persistence_typed_round_trip(writer, label="x5-sqlite-file")
    del writer

    reader = _sqlite_problem_persistence(db_path)
    assert_problem_persistence_conformance(reader, label="x5-sqlite-file-reopen")


def test_sqlite_file_tenant_isolation_on_fresh_open(tmp_path: Path) -> None:
    db_path = tmp_path / "tenant.sqlite3"
    persistence = _sqlite_problem_persistence(db_path)
    assert_problem_persistence_conformance(persistence, label="x5-sqlite-tenant")
