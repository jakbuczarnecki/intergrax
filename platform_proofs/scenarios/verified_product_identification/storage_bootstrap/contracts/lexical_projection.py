"""Typed lexical index projection contracts — derived retrieval state only."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from platform_proofs.scenarios.verified_product_identification.application.domain.source import (
    SourceRecordRef,
)


@dataclass(frozen=True, slots=True)
class LexicalIndexProjectionRecord:
    """One derived lexical document bound to immutable source identity."""

    source_ref: SourceRecordRef
    lexical_document: str
    document_hash: str
    document_length: int
    derivation_version: str

    def __post_init__(self) -> None:
        if not self.lexical_document.strip():
            raise ValueError("LexicalIndexProjectionRecord.lexical_document must be non-empty")
        if not self.document_hash.strip():
            raise ValueError("LexicalIndexProjectionRecord.document_hash must be non-empty")
        if self.document_length < 1:
            raise ValueError("LexicalIndexProjectionRecord.document_length must be >= 1")
        if not self.derivation_version.strip():
            raise ValueError("LexicalIndexProjectionRecord.derivation_version must be non-empty")


def lexical_document_hash(*, lexical_document: str, derivation_version: str) -> str:
    payload = f"{lexical_document}\0{derivation_version}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
