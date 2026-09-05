"""Deterministic query benchmark construction from arena sample records."""

from __future__ import annotations

from platform_proofs.scenarios.verified_product_identification.arena.contracts.classification import (
    QueryDifficultyClass,
)
from platform_proofs.scenarios.verified_product_identification.arena.contracts.query_benchmark import (
    ArenaSourceRef,
    BenchmarkClusterEvidence,
    EmbeddingArenaQueryCase,
)
from platform_proofs.scenarios.verified_product_identification.arena.sampling.arena_sample import (
    ArenaSampleRecord,
)

LONG_DESCRIPTION_QUERY_THRESHOLD = 240


def _source_ref(record: ArenaSampleRecord) -> ArenaSourceRef:
    return ArenaSourceRef(
        offer_id=record.offer_id,
        global_row_index=record.global_row_index,
    )


def _cluster_evidence(record: ArenaSampleRecord) -> BenchmarkClusterEvidence | None:
    if record.source_offer.cluster_id is None:
        return None
    return BenchmarkClusterEvidence(
        cluster_id=record.source_offer.cluster_id,
        purpose="benchmark_only_cluster_reference",
    )


def _select_hard_negatives(
    target: ArenaSampleRecord,
    records: tuple[ArenaSampleRecord, ...],
    *,
    limit: int = 5,
) -> tuple[str, ...]:
    negatives: list[str] = []
    target_title = (target.source_offer.title or "").casefold()
    target_brand = (target.source_offer.brand or "").casefold()
    for candidate in records:
        if candidate.offer_id == target.offer_id:
            continue
        score = 0
        candidate_title = (candidate.source_offer.title or "").casefold()
        candidate_brand = (candidate.source_offer.brand or "").casefold()
        if target_title and candidate_title and target_title[:16] == candidate_title[:16]:
            score += 2
        if target_brand and candidate_brand and target_brand == candidate_brand:
            score += 2
        if (
            target.source_offer.cluster_id is not None
            and candidate.source_offer.cluster_id == target.source_offer.cluster_id
            and candidate.offer_id != target.offer_id
        ):
            score += 1
        if score > 0:
            negatives.append((score, candidate.offer_id))
    negatives.sort(key=lambda item: (-item[0], item[1]))
    return tuple(offer_id for _, offer_id in negatives[:limit])


def _build_strong_identity_query(record: ArenaSampleRecord) -> str | None:
    if not record.source_offer.identifiers:
        return None
    identifier = record.source_offer.identifiers[0]
    title_fragment = ""
    if record.source_offer.title:
        title_fragment = record.source_offer.title.split()[0]
    return f"{identifier.source_value} {title_fragment}".strip()


def _build_title_brand_query(record: ArenaSampleRecord) -> str | None:
    if not record.source_offer.title:
        return None
    if record.source_offer.brand:
        return f"{record.source_offer.brand} {record.source_offer.title}"
    return None


def _build_title_only_query(record: ArenaSampleRecord) -> str | None:
    if not record.source_offer.title:
        return None
    words = record.source_offer.title.split()
    if len(words) <= 2:
        return record.source_offer.title
    return " ".join(words[: max(2, len(words) // 2)])


def _build_structured_attributes_query(record: ArenaSampleRecord) -> str | None:
    if not record.source_offer.key_value_pairs:
        return None
    fragments = [
        pair.source_value
        for pair in record.source_offer.key_value_pairs[:3]
        if pair.source_value.strip()
    ]
    if not fragments:
        return None
    return " ".join(fragments)


def _build_partial_noisy_query(record: ArenaSampleRecord) -> str | None:
    if not record.source_offer.title:
        return None
    words = [word for word in record.source_offer.title.split() if word]
    if len(words) < 3:
        return None
    return " ".join(words[::2])


def _build_long_description_query(record: ArenaSampleRecord) -> str | None:
    description = record.source_offer.description
    if description is None or len(description) < LONG_DESCRIPTION_QUERY_THRESHOLD:
        return None
    words = description.split()
    if len(words) < 12:
        return description[:180]
    return " ".join(words[:12])


def build_query_benchmark_cases(
    records: tuple[ArenaSampleRecord, ...],
    *,
    max_cases: int = 250,
) -> tuple[EmbeddingArenaQueryCase, ...]:
    if not records:
        msg = "records must not be empty"
        raise ValueError(msg)

    builders: tuple[tuple[QueryDifficultyClass, str], ...] = (
        (QueryDifficultyClass.STRONG_IDENTITY, "strong_identity"),
        (QueryDifficultyClass.TITLE_BRAND, "title_brand"),
        (QueryDifficultyClass.TITLE_ONLY, "title_only"),
        (QueryDifficultyClass.STRUCTURED_ATTRIBUTES, "structured_attributes"),
        (QueryDifficultyClass.PARTIAL_NOISY, "partial_noisy"),
        (QueryDifficultyClass.LONG_DESCRIPTION_SIGNAL, "long_description_signal"),
    )

    cases: list[EmbeddingArenaQueryCase] = []
    case_counter = 0
    for record in records:
        for difficulty, builder_name in builders:
            if len(cases) >= max_cases:
                return tuple(cases)
            query_text: str | None
            if builder_name == "strong_identity":
                query_text = _build_strong_identity_query(record)
            elif builder_name == "title_brand":
                query_text = _build_title_brand_query(record)
            elif builder_name == "title_only":
                query_text = _build_title_only_query(record)
            elif builder_name == "structured_attributes":
                query_text = _build_structured_attributes_query(record)
            elif builder_name == "partial_noisy":
                query_text = _build_partial_noisy_query(record)
            else:
                query_text = _build_long_description_query(record)
            if query_text is None or not query_text.strip():
                continue
            if "cluster_id" in query_text:
                continue
            case_counter += 1
            cases.append(
                EmbeddingArenaQueryCase(
                    case_id=f"q-{case_counter:04d}",
                    query_text=query_text.strip(),
                    difficulty=difficulty,
                    relevant_source_refs=(_source_ref(record),),
                    provenance=f"{builder_name} from offer_id={record.offer_id}",
                    benchmark_only_cluster_evidence=_cluster_evidence(record),
                    hard_negative_offer_ids=_select_hard_negatives(record, records),
                    is_long_input_query=len(query_text) >= LONG_DESCRIPTION_QUERY_THRESHOLD,
                )
            )
    return tuple(cases)
