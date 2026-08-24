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


def test_observability_public_exports_include_schema_and_persistence() -> None:
    from intergrax.runtime import observability

    assert "APPLICATION_OBSERVABILITY_ATTRIBUTES_SCHEMA" in observability.__all__
    assert "CausalEvidencePersistence" in observability.__all__

    from intergrax.runtime.observability import (
        APPLICATION_OBSERVABILITY_ATTRIBUTES_SCHEMA,
        CausalEvidencePersistence,
    )

    assert APPLICATION_OBSERVABILITY_ATTRIBUTES_SCHEMA
    assert CausalEvidencePersistence is not None


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
