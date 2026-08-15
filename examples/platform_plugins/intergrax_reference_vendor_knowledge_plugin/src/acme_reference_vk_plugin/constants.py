"""Stable identities for the reference external provider."""

from __future__ import annotations

ACME_REFERENCE_PROVIDER_ID = "acme_reference"
ACME_DOCUMENTS_SOURCE_KIND = "acme_documents"
ACME_COLLECTION_SCOPE_TYPE = "acme_reference.collection.v1"
ACME_REFERENCE_MARKER = "acme-reference-marker-42"
ACME_STRUCTURED_RECORD_SCHEMA = "application/vnd.intergrax.acme-reference-document+json"
ACME_DEFAULT_COLLECTION_ID = "col-ref-qualification-1"
ACME_ADAPTER_RUNTIME_REF = (
    "knowledge-adapter:acme_reference:wiki_knowledge:acme_documents"
)
ACME_INDEXED_RUNTIME_REF = "indexed-source:acme_reference:acme_documents"
