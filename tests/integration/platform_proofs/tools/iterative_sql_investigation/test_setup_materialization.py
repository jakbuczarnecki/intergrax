# © Artur Czarnecki. All rights reserved.

"""Integration tests for canonical dataset setup and materialization."""

from __future__ import annotations

import os

import pytest

from platform_proofs.tools.iterative_sql_investigation.dataset import PROOF_ROW_COUNT
from platform_proofs.tools.iterative_sql_investigation.dataset_identity import (
    compute_dataset_fingerprint,
)
from platform_proofs.tools.iterative_sql_investigation.setup import (
    DatasetSetupError,
    materialize_and_verify_dataset,
)
from platform_proofs.tools.iterative_sql_investigation.runtime import ADMIN_DSN_ENV, DEFAULT_ADMIN_DSN

pytestmark = [pytest.mark.integration, pytest.mark.network]


def _admin_available() -> bool:
    dsn = os.environ.get(ADMIN_DSN_ENV, DEFAULT_ADMIN_DSN).strip()
    if not dsn:
        return False
    try:
        from intergrax.integrations.providers.relational_store.postgresql import (
            create_postgresql_relational_store,
        )

        store = create_postgresql_relational_store(dsn=dsn, tenant_schema="proof")
        store.connect()
        store.fetch_all("SELECT 1 AS ok")
        store.close()
        return True
    except Exception:
        return False


@pytest.fixture(scope="module")
def require_proof_postgres() -> None:
    if not _admin_available():
        pytest.skip("Proof PostgreSQL admin backend unavailable for setup integration")


def test_materialized_db_has_exactly_100k_rows(require_proof_postgres: None) -> None:
    dsn = os.environ.get(ADMIN_DSN_ENV, DEFAULT_ADMIN_DSN)
    result = materialize_and_verify_dataset(admin_dsn=dsn)
    assert result.loaded_rows == PROOF_ROW_COUNT
    assert result.db_stats.total_rows == PROOF_ROW_COUNT
    assert result.verified is True


def test_setup_fingerprint_matches_canonical(require_proof_postgres: None) -> None:
    dsn = os.environ.get(ADMIN_DSN_ENV, DEFAULT_ADMIN_DSN)
    result = materialize_and_verify_dataset(admin_dsn=dsn)
    expected = compute_dataset_fingerprint(result.identity)
    assert result.fingerprint.sha256 == expected.sha256


def test_db_observed_invariants_pass(require_proof_postgres: None) -> None:
    dsn = os.environ.get(ADMIN_DSN_ENV, DEFAULT_ADMIN_DSN)
    result = materialize_and_verify_dataset(admin_dsn=dsn)
    assert result.contract_checks["north_worse_than_non_north"] is True
    assert result.contract_checks["anomaly_segment_rate_above_half"] is True
    assert result.contract_checks["row_count_exact"] is True
