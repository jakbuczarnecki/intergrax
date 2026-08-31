# © Artur Czarnecki. All rights reserved.

"""DIAG-ENTERPRISE-1 real Mongo scalable Problem list query proof."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from intergrax.runtime.diagnostics.diagnostic_subject import DiagnosticSubjectKind
from intergrax.runtime.diagnostics.lifecycle_analysis import LifecycleAnomalyKind
from intergrax.runtime.diagnostics.persistence_conformance import (
    _sample_reconciliation_key,
    _sample_signature,
    _sample_subject_ref,
    sample_problem,
)
from intergrax.runtime.diagnostics.problem_grouping import (
    DeterministicLimitationSignature,
    DeterministicProblemSignature,
    DiagnosticLimitationKind,
)
from intergrax.runtime.diagnostics.problem_lifecycle import ProblemStatus, mint_problem_id
from tests.integration.applications.architecture.harden_4f_mongo_support import (
    cleanup_proof_tenant,
    create_proof_document_store,
    ensure_mongo_running,
    require_docker_for_harden_4f_proof,
)
from tests.unit.runtime.diagnostics.problem_persistence_test_support import (
    document_store_problem_persistence_for_tests,
)

_TENANT = "diag-enterprise-1-mongo"
_BASE_TIME = datetime(2026, 8, 26, 9, 0, tzinfo=UTC)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.external_proof,
    pytest.mark.network,
    pytest.mark.no_ci,
]


@pytest.fixture
def mongo_enterprise_read_lifecycle(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    from tests.integration.applications.architecture.harden_4f_mongo_support import proof_env

    env = proof_env()
    monkeypatch.setenv("INTERGRAX_MONGODB_URI", env["INTERGRAX_MONGODB_URI"])
    monkeypatch.setenv("INTERGRAX_MONGODB_DATABASE", env["INTERGRAX_MONGODB_DATABASE"])
    monkeypatch.setenv("INTERGRAX_MONGODB_COLLECTION", env["INTERGRAX_MONGODB_COLLECTION"])
    require_docker_for_harden_4f_proof()
    ensure_mongo_running()
    cleanup_proof_tenant(tenant_id=_TENANT)
    try:
        yield
    finally:
        cleanup_proof_tenant(tenant_id=_TENANT)
        ensure_mongo_running()


def test_mongo_paginated_problem_query_order_and_status(
    mongo_enterprise_read_lifecycle: None,
) -> None:
    del mongo_enterprise_read_lifecycle
    store = create_proof_document_store()
    try:
        persistence = document_store_problem_persistence_for_tests(store)
        anomaly_kinds = list(LifecycleAnomalyKind)
        for index in range(12):
            signature = DeterministicProblemSignature(
                findings=_sample_signature().findings,
                limitations=(
                    DeterministicLimitationSignature(
                        kind=DiagnosticLimitationKind.RUNTIME_HISTORY_TRUNCATED,
                        source_anomaly_kind=anomaly_kinds[index % len(anomaly_kinds)],
                    ),
                ),
                subject_domain=(
                    DiagnosticSubjectKind.EXECUTION
                    if index < len(anomaly_kinds)
                    else DiagnosticSubjectKind.APPLICATION_INSTANCE
                ),
            )
            problem = replace(
                sample_problem(
                    tenant_id=_TENANT,
                    problem_id=mint_problem_id(),
                    subject_refs=(_sample_subject_ref(tenant_id=_TENANT),),
                    reconciliation_key=_sample_reconciliation_key(
                        tenant_id=_TENANT,
                        signature=signature,
                    ),
                ),
                last_seen_at=_BASE_TIME + timedelta(minutes=index),
                first_seen_at=_BASE_TIME,
                status=ProblemStatus.OPEN if index % 2 == 0 else ProblemStatus.RESOLVED,
            )
            persistence.create(problem)

        page1 = persistence.query_problems(tenant_id=_TENANT, limit=5)
        page2 = persistence.query_problems(
            tenant_id=_TENANT,
            limit=5,
            cursor=page1.next_cursor,
        )
        open_page = persistence.query_problems(
            tenant_id=_TENANT,
            status=ProblemStatus.OPEN,
            limit=10,
        )

        combined = [*page1.problems, *page2.problems]
        assert len(page1.problems) == 5
        assert page1.has_more is True
        assert page1.next_cursor is not None
        assert len({item.problem_id for item in combined}) == 10
        assert combined[0].last_seen_at >= combined[-1].last_seen_at
        assert all(item.status is ProblemStatus.OPEN for item in open_page.problems)
        assert len(open_page.problems) == 6
    finally:
        store.close()


def test_mongo_orphan_list_index_is_skipped_without_integrity_error(
    mongo_enterprise_read_lifecycle: None,
) -> None:
    del mongo_enterprise_read_lifecycle
    store = create_proof_document_store()
    try:
        persistence = document_store_problem_persistence_for_tests(store)
        problem = sample_problem(tenant_id=_TENANT, problem_id=mint_problem_id())
        persistence.create(problem)
        store.delete(
            f"intergrax.diagnostic_problem.v1:{_TENANT}",
            f"record:{problem.problem_id}",
        )
        page = persistence.query_problems(tenant_id=_TENANT, limit=5)
        assert page.problems == ()
        assert page.has_more is False
    finally:
        store.close()
