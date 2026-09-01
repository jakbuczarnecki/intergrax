# © Artur Czarnecki. All rights reserved.

"""PROVIDER-QUAL-5 — real MongoDB validity persistence reopen proof."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace
from datetime import datetime, timezone

import pytest

from intergrax.core.qualification import QualificationEvidenceValidity
from intergrax.core.qualification.validity_persistence import (
    DocumentStoreProviderQualificationValidityPersistence,
)
from intergrax.core.qualification import (
    new_validity_evaluation_id,
    record_provider_qualification_validity_revocation,
)
from intergrax.integrations.providers.document_store.mongodb.adapter import _MongoDBDocumentStore
from tests.integration.applications.architecture.harden_4f_mongo_support import (
    create_proof_document_store,
    ensure_mongo_running,
    require_docker_for_harden_4f_proof,
)
from tests.unit.core.qualification.test_provider_qualification_validity_persistence import (
    _current_record,
    _stale_record,
)
from tests.unit.core.qualification.test_provider_qualification_persistence import _run

pytestmark = [
    pytest.mark.integration,
    pytest.mark.external_proof,
    pytest.mark.network,
    pytest.mark.no_ci,
]

_EVALUATED_AT_T2 = datetime(2026, 8, 18, 9, 0, 0, tzinfo=timezone.utc)
_EVALUATED_AT_T3 = datetime(2026, 8, 19, 10, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def mongo_qualification_validity_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
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


def test_provider_qualification_validity_survives_real_document_store_reopen(
    mongo_qualification_validity_env: None,
) -> None:
    del mongo_qualification_validity_env
    run = _run()
    current = _current_record(run.qualification_run_id)
    stale = _stale_record(run.qualification_run_id)

    store_a = create_proof_document_store()
    client_a = _mongo_collection_client(store_a)
    persistence_a = DocumentStoreProviderQualificationValidityPersistence(store_a)
    persistence_a.append_evaluation(current)
    persistence_a.append_evaluation(stale)
    store_a.close()
    del persistence_a

    store_b = create_proof_document_store()
    client_b = _mongo_collection_client(store_b)
    assert store_a is not store_b
    assert client_a is not client_b

    persistence_b = DocumentStoreProviderQualificationValidityPersistence(store_b)
    try:
        history = persistence_b.list_evaluations(run.qualification_run_id)
        interpretation = persistence_b.get_current_validity(run.qualification_run_id)
        assert len(history) == 2
        assert history[0] == current
        assert history[1] == stale
        assert interpretation is not None
        assert interpretation.validity is QualificationEvidenceValidity.STALE
    finally:
        store_b.close()


def test_terminal_revocation_survives_real_document_store_reopen(
    mongo_qualification_validity_env: None,
) -> None:
    del mongo_qualification_validity_env
    run = _run()
    current = _current_record(run.qualification_run_id)
    revoked = record_provider_qualification_validity_revocation(
        run.qualification_run_id,
        reason="operator_revoked",
        evaluated_at=_EVALUATED_AT_T2,
        validity_evaluation_id=new_validity_evaluation_id(),
    )
    later_current = replace(
        current,
        evaluated_at=_EVALUATED_AT_T3,
        validity_evaluation_id=new_validity_evaluation_id(),
    )

    store_a = create_proof_document_store()
    persistence_a = DocumentStoreProviderQualificationValidityPersistence(store_a)
    persistence_a.append_evaluation(current)
    persistence_a.append_evaluation(revoked)
    persistence_a.append_evaluation(later_current)
    store_a.close()
    del persistence_a

    store_b = create_proof_document_store()
    persistence_b = DocumentStoreProviderQualificationValidityPersistence(store_b)
    try:
        history = persistence_b.list_evaluations(run.qualification_run_id)
        interpretation = persistence_b.get_current_validity(run.qualification_run_id)
        assert len(history) == 3
        assert history[0] == current
        assert history[1] == revoked
        assert history[2] == later_current
        assert interpretation is not None
        assert interpretation.validity is QualificationEvidenceValidity.REVOKED
    finally:
        store_b.close()
