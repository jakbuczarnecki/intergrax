# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Pluggable functional evidence persistence providers (Evidence Plane)."""

from intergrax.runtime.observability.functional_evidence.document_store_functional_evidence_persistence import (
    DocumentStoreFunctionalEvidencePersistence,
    build_document_store_functional_evidence_persistence,
    wire_functional_evidence_persistence,
)
from intergrax.runtime.observability.functional_evidence.functional_evidence_persistence_conformance import (
    assert_functional_evidence_persistence_conformance,
    sample_functional_evidence,
)
from intergrax.runtime.observability.functional_evidence.in_memory_functional_evidence_persistence import (
    InMemoryFunctionalEvidencePersistence,
    build_in_memory_functional_evidence_persistence,
)

__all__ = [
    "DocumentStoreFunctionalEvidencePersistence",
    "InMemoryFunctionalEvidencePersistence",
    "assert_functional_evidence_persistence_conformance",
    "build_document_store_functional_evidence_persistence",
    "build_in_memory_functional_evidence_persistence",
    "sample_functional_evidence",
    "wire_functional_evidence_persistence",
]
