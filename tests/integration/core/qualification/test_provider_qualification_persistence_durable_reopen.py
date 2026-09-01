# © Artur Czarnecki. All rights reserved.

"""PROVIDER-QUAL-3C-R2 — real durable DocumentStore reopen proof for qualification evidence."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from intergrax.core.qualification.persistence import DocumentStoreProviderQualificationPersistence
from intergrax.integrations.providers.document_store.mongodb.adapter import _MongoDBDocumentStore
from tests.integration.applications.architecture.harden_4f_mongo_support import (
    create_proof_document_store,
    ensure_mongo_running,
    require_docker_for_harden_4f_proof,
)
from tests.unit.core.qualification.test_provider_qualification_persistence import _run

pytestmark = [
    pytest.mark.integration,
    pytest.mark.external_proof,
    pytest.mark.network,
    pytest.mark.no_ci,
]


@pytest.fixture
def mongo_qualification_persistence_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
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


def test_provider_qualification_run_survives_real_document_store_reopen(
    mongo_qualification_persistence_env: None,
) -> None:
    """
    Real MongoDB durability proof:

    store A -> persist -> close -> discard adapter A -> store B -> recover same run.

    No shared fake collection or in-process Python dict backs persistence.
    """
    del mongo_qualification_persistence_env
    run = _run()

    store_a = create_proof_document_store()
    client_a = _mongo_collection_client(store_a)
    persistence_a = DocumentStoreProviderQualificationPersistence(store_a)
    persistence_a.persist(run)
    store_a.close()
    del persistence_a

    store_b = create_proof_document_store()
    client_b = _mongo_collection_client(store_b)
    assert store_a is not store_b
    assert client_a is not client_b

    persistence_b = DocumentStoreProviderQualificationPersistence(store_b)
    try:
        loaded = persistence_b.get_by_qualification_run_id(run.qualification_run_id)
        assert loaded == run
    finally:
        store_b.close()
