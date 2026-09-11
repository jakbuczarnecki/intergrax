"""Okapi BM25 scoring for indexed lexical retrieval."""

from __future__ import annotations

import math
from dataclasses import dataclass

from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.adapters.postgresql.lexical_tokenization import (
    tokenize_lexical_document,
)


@dataclass(frozen=True, slots=True)
class Bm25CorpusStatistics:
    document_count: int
    average_document_length: float


@dataclass(frozen=True, slots=True)
class Bm25IndexedDocument:
    catalog_id: str
    offer_id: str
    source_revision_norm: str
    source_revision: str | None
    document_length: int
    term_frequencies: dict[str, int]


@dataclass(frozen=True, slots=True)
class Bm25ScoredDocument:
    catalog_id: str
    offer_id: str
    source_revision_norm: str
    bm25_score: float


BM25_K1: float = 1.2
BM25_B: float = 0.75


@dataclass(frozen=True, slots=True)
class Bm25EngineConfiguration:
    k1: float = BM25_K1
    b: float = BM25_B


def build_term_frequencies(text: str) -> tuple[tuple[str, ...], dict[str, int]]:
    tokens = tokenize_lexical_document(text)
    frequencies: dict[str, int] = {}
    for token in tokens:
        frequencies[token] = frequencies.get(token, 0) + 1
    return tokens, frequencies


def compute_bm25_score(
    *,
    query_terms: tuple[str, ...],
    document: Bm25IndexedDocument,
    corpus: Bm25CorpusStatistics,
    term_document_frequencies: dict[str, int],
    configuration: Bm25EngineConfiguration,
) -> float:
    if corpus.document_count <= 0 or not query_terms:
        return 0.0

    score = 0.0
    average_length = max(corpus.average_document_length, 1.0)
    for term in query_terms:
        term_frequency = document.term_frequencies.get(term)
        if term_frequency is None or term_frequency <= 0:
            continue
        document_frequency = term_document_frequencies.get(term, 0)
        if document_frequency <= 0:
            continue
        idf = math.log(
            1.0
            + (corpus.document_count - document_frequency + 0.5)
            / (document_frequency + 0.5)
        )
        length_norm = configuration.k1 * (
            1.0 - configuration.b + configuration.b * document.document_length / average_length
        )
        numerator = term_frequency * (configuration.k1 + 1.0)
        denominator = term_frequency + length_norm
        score += idf * numerator / max(denominator, 1e-9)
    return score


def rank_bm25_documents(
    *,
    query_text: str,
    documents: tuple[Bm25IndexedDocument, ...],
    corpus: Bm25CorpusStatistics,
    term_document_frequencies: dict[str, int],
    limit: int,
    configuration: Bm25EngineConfiguration | None = None,
) -> tuple[Bm25ScoredDocument, ...]:
    resolved = configuration or Bm25EngineConfiguration()
    query_terms = tokenize_lexical_document(query_text)
    if not query_terms or not documents:
        return ()

    scored: list[Bm25ScoredDocument] = []
    for document in documents:
        bm25_score = compute_bm25_score(
            query_terms=query_terms,
            document=document,
            corpus=corpus,
            term_document_frequencies=term_document_frequencies,
            configuration=resolved,
        )
        if bm25_score > 0.0:
            scored.append(
                Bm25ScoredDocument(
                    catalog_id=document.catalog_id,
                    offer_id=document.offer_id,
                    source_revision_norm=document.source_revision_norm,
                    bm25_score=bm25_score,
                )
            )

    scored.sort(
        key=lambda item: (
            -item.bm25_score,
            item.catalog_id,
            item.offer_id,
            item.source_revision_norm,
        )
    )
    return tuple(scored[:limit])
