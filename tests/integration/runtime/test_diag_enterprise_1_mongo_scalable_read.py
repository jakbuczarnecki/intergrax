# © Artur Czarnecki. All rights reserved.

"""DIAG-ENTERPRISE-1 real Mongo scalable Problem list query proof."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta

from dataclasses import replace

import pytest

from intergrax.runtime.diagnostics.document_store_problem_persistence import (
    DocumentStoreProblemPersistence,
)
from intergrax.runtime.diagnostics.persistence_conformance import sample_problem
from intergrax.runtime.diagnostics.problem_lifecycle import ProblemStatus, mint_problem_id
from intergrax.runtime.diagnostics.problem_persistence import ProblemPersistenceIntegrityError
from tests.integration.applications.architecture.harden_4f_mongo_support import (
    cleanup_proof_tenant,
    create_proof_document_store,
    ensure_mongo_running,
    require_docker_for_harden_4f_proof,
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
def mongo_enterprise_read_lifecycle() -> Iterator[None]:
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
        persistence = DocumentStoreProblemPersistence(store)
        for index in range(12):
            problem = replace(
                sample_problem(
                    tenant_id=_TENANT,
                    problem_id=mint_problem_id(),
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


def test_mongo_list_index_integrity_surfaces_missing_canonical(
    mongo_enterprise_read_lifecycle: None,
) -> None:
    del mongo_enterprise_read_lifecycle
    store = create_proof_document_store()
    try:
        persistence = DocumentStoreProblemPersistence(store)
        problem = sample_problem(tenant_id=_TENANT, problem_id=mint_problem_id())
        persistence.create(problem)
        store.delete(
            f"intergrax.diagnostic_problem.v1:{_TENANT}",
            f"record:{problem.problem_id}",
        )
        with pytest.raises(
            ProblemPersistenceIntegrityError,
            match="canonical Problem record missing for list index",
        ):
            persistence.query_problems(tenant_id=_TENANT, limit=5)
    finally:
        store.close()
