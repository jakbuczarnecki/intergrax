# © Artur Czarnecki. All rights reserved.

"""PROVIDER-QUAL-4 — real MongoDB discovery proof over persisted qualification evidence."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime, timedelta, timezone

import pytest

from intergrax.core.qualification.discovery import ProviderQualificationRunFilter
from intergrax.core.qualification.persistence import DocumentStoreProviderQualificationPersistence
from intergrax.core.qualification.validity import QualificationRunId
from intergrax.integrations.providers.document_store.mongodb.adapter import _MongoDBDocumentStore
from tests.integration.applications.architecture.harden_4f_mongo_support import (
    create_proof_document_store,
    ensure_mongo_running,
    require_docker_for_harden_4f_proof,
)
from tests.unit.core.qualification.test_provider_qualification_discovery import _run

pytestmark = [
    pytest.mark.integration,
    pytest.mark.external_proof,
    pytest.mark.network,
    pytest.mark.no_ci,
]


@pytest.fixture
def mongo_qualification_discovery_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    from tests.integration.applications.architecture.harden_4f_mongo_support import proof_env

    env = proof_env()
    monkeypatch.setenv("INTERGRAX_MONGODB_URI", env["INTERGRAX_MONGODB_URI"])
    monkeypatch.setenv("INTERGRAX_MONGODB_DATABASE", env["INTERGRAX_MONGODB_DATABASE"])
    monkeypatch.setenv("INTERGRAX_MONGODB_COLLECTION", env["INTERGRAX_MONGODB_COLLECTION"])
    require_docker_for_harden_4f_proof()
    ensure_mongo_running()
    yield


def _mongo_collection_client(store: object) -> object:
    assert isinstance(store, _MongoDBDocumentStore)
    return store.mongo_client


def test_provider_qualification_discovery_over_real_mongodb_reopen(
    mongo_qualification_discovery_env: None,
) -> None:
    del mongo_qualification_discovery_env
    base_time = datetime(2026, 8, 17, 12, 0, 0, tzinfo=timezone.utc)
    postgresql_run = _run(
        run_id=QualificationRunId("qual_run_11111111111111111111111111111111"),
        provider_id="postgresql",
        executed_at=base_time,
    )
    sqlite_run = _run(
        run_id=QualificationRunId("qual_run_22222222222222222222222222222222"),
        provider_id="sqlite",
        executed_at=base_time + timedelta(hours=1),
    )

    store_a = create_proof_document_store()
    client_a = _mongo_collection_client(store_a)
    persistence_a = DocumentStoreProviderQualificationPersistence(store_a)
    persistence_a.persist(postgresql_run)
    persistence_a.persist(sqlite_run)
    store_a.close()
    del persistence_a

    store_b = create_proof_document_store()
    client_b = _mongo_collection_client(store_b)
    assert store_a is not store_b
    assert client_a is not client_b

    persistence_b = DocumentStoreProviderQualificationPersistence(store_b)
    try:
        pg_page = persistence_b.find_runs(
            ProviderQualificationRunFilter(provider_id="postgresql"),
        )
        capability_page = persistence_b.find_runs(
            ProviderQualificationRunFilter(
                capability_id="collaborative_work.persistence.v1",
            ),
        )
    finally:
        store_b.close()

    assert [item.qualification_run_id for item in pg_page.runs] == [
        postgresql_run.qualification_run_id,
    ]
    assert {item.qualification_run_id for item in capability_page.runs} == {
        postgresql_run.qualification_run_id,
        sqlite_run.qualification_run_id,
    }


def test_provider_qualification_discovery_paginates_over_real_mongodb(
    mongo_qualification_discovery_env: None,
) -> None:
    del mongo_qualification_discovery_env
    base_time = datetime(2026, 8, 17, 12, 0, 0, tzinfo=timezone.utc)
    runs = tuple(
        _run(
            run_id=QualificationRunId(f"qual_run_{index:032x}"),
            provider_id="postgresql",
            executed_at=base_time + timedelta(minutes=index),
        )
        for index in range(5)
    )

    store = create_proof_document_store()
    persistence = DocumentStoreProviderQualificationPersistence(store)
    try:
        for item in runs:
            persistence.persist(item)

        first = persistence.find_runs(
            ProviderQualificationRunFilter(provider_id="postgresql"),
            limit=2,
        )
        second = persistence.find_runs(
            ProviderQualificationRunFilter(provider_id="postgresql"),
            limit=2,
            cursor=first.next_cursor,
        )
        third = persistence.find_runs(
            ProviderQualificationRunFilter(provider_id="postgresql"),
            limit=2,
            cursor=second.next_cursor,
        )
    finally:
        store.close()

    assert len(first.runs) == 2
    assert len(second.runs) == 2
    assert len(third.runs) == 1
    assert first.next_cursor is not None
    assert second.next_cursor is not None
    assert third.next_cursor is None
    discovered = first.runs + second.runs + third.runs
    assert len(discovered) == len({item.qualification_run_id for item in discovered}) == 5
    assert [item.qualification_run_id for item in discovered] == [
        runs[4].qualification_run_id,
        runs[3].qualification_run_id,
        runs[2].qualification_run_id,
        runs[1].qualification_run_id,
        runs[0].qualification_run_id,
    ]
