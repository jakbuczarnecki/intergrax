# © Artur Czarnecki. All rights reserved.

"""DIAG-1P1 causal evidence persistence conformance — shared contract across backends."""

from __future__ import annotations

import pytest

from intergrax.runtime.observability.causal_evidence_persistence import (
    CausalEvidencePersistence,
)
from intergrax.runtime.observability.memory_causal_evidence_persistence import (
    InMemoryCausalEvidencePersistence,
)
from intergrax.runtime.observability.persistence_conformance import (
    assert_causal_evidence_persistence_conformance,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("label", "factory"),
    [
        ("memory", lambda: InMemoryCausalEvidencePersistence()),
    ],
)
def test_causal_evidence_persistence_conformance_matrix(
    label: str,
    factory,
) -> None:
    store: CausalEvidencePersistence = factory()
    try:
        assert_causal_evidence_persistence_conformance(store, label=label)
    finally:
        store.close()
