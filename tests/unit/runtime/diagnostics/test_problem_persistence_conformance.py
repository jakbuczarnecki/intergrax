# © Artur Czarnecki. All rights reserved.

"""DIAG-STORAGE Problem persistence conformance — shared contract across backends."""

from __future__ import annotations

import pytest

from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.runtime.diagnostics.document_store_problem_persistence import (
    DocumentStoreProblemPersistence,
)
from intergrax.runtime.diagnostics.in_memory_problem_persistence import (
    InMemoryProblemPersistence,
)
from intergrax.runtime.diagnostics.persistence_conformance import (
    assert_problem_persistence_conformance,
    assert_problem_persistence_typed_round_trip,
    assert_problem_update_publishes_subject_indexes_atomically,
)
from intergrax.runtime.diagnostics.problem_persistence import ProblemPersistence

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("label", "factory"),
    [
        ("memory", lambda: InMemoryProblemPersistence()),
        (
            "document_store",
            lambda: DocumentStoreProblemPersistence(InMemoryDocumentStore()),
        ),
    ],
)
def test_problem_persistence_conformance_matrix(label: str, factory) -> None:
    store: ProblemPersistence = factory()
    try:
        assert_problem_persistence_conformance(store, label=label)
        assert_problem_update_publishes_subject_indexes_atomically(store, label=label)
        assert_problem_persistence_typed_round_trip(store, label=label)
    finally:
        store.close()
