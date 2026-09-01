# © Artur Czarnecki. All rights reserved.

"""DIAG-ENTERPRISE-2 ProblemOccurrence persistence conformance."""

from __future__ import annotations

import pytest

from intergrax.runtime.diagnostics.persistence_conformance import (
    assert_problem_occurrence_persistence_conformance,
)
from intergrax.runtime.diagnostics.problem_occurrence_persistence import (
    ProblemOccurrencePersistence,
)
from tests.unit.runtime.diagnostics.problem_persistence_test_support import (
    document_store_occurrence_persistence_for_tests,
    in_memory_document_store_for_problem_tests,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("label", "factory"),
    [
        (
            "document_store",
            lambda: document_store_occurrence_persistence_for_tests(
                in_memory_document_store_for_problem_tests(),
            ),
        ),
    ],
)
def test_problem_occurrence_persistence_conformance_matrix(
    label: str,
    factory,
) -> None:
    store: ProblemOccurrencePersistence = factory()
    assert_problem_occurrence_persistence_conformance(store, label=label)
