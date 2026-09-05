"""Deterministic arena sample construction from WDC-derived strata."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from platform_proofs.scenarios.verified_product_identification.application.domain.wdc_source_offer import (
    WdcSourceOffer,
)
from platform_proofs.scenarios.verified_product_identification.arena.contracts.results import (
    ArenaSampleManifest,
    ArenaSampleRecordSnapshot,
)
from platform_proofs.scenarios.verified_product_identification.arena.contracts.versioning import (
    ARENA_SAMPLE_VERSION,
)

ARENA_SAMPLE_TARGET_SIZE = 1000
ARENA_SAMPLE_SCAN_ROW_LIMIT = 50_000
ARENA_SAMPLE_SELECTION_SEED = "vpi-embedding-arena-v1"

STRATA_QUOTAS: tuple[tuple[str, int], ...] = (
    ("has_identifiers", 120),
    ("no_identifiers", 80),
    ("has_brand", 120),
    ("missing_brand", 80),
    ("long_description", 100),
    ("short_description", 100),
    ("has_key_value_pairs", 100),
    ("has_spec_table_content", 80),
    ("multi_offer_cluster", 80),
    ("singleton_cluster", 80),
    ("strong_title", 80),
    ("near_similar_title", 80),
)


@dataclass(frozen=True, slots=True)
class ArenaSampleRecord:
    offer_id: str
    global_row_index: int
    semantic_text: str
    source_offer: WdcSourceOffer
    strata_tags: tuple[str, ...]


def _description_length_class(description: str | None) -> str:
    if description is None:
        return "short_description"
    return "long_description" if len(description) >= 240 else "short_description"


def derive_strata_tags(source_offer: WdcSourceOffer) -> tuple[str, ...]:
    tags: list[str] = []
    if source_offer.identifiers:
        tags.append("has_identifiers")
    else:
        tags.append("no_identifiers")
    if source_offer.brand:
        tags.append("has_brand")
    else:
        tags.append("missing_brand")
    tags.append(_description_length_class(source_offer.description))
    if source_offer.key_value_pairs:
        tags.append("has_key_value_pairs")
    if source_offer.spec_table_content:
        tags.append("has_spec_table_content")
    if source_offer.cluster_id is not None and source_offer.cluster_id > 0:
        tags.append("multi_offer_cluster")
    else:
        tags.append("singleton_cluster")
    if source_offer.title and len(source_offer.title) >= 24:
        tags.append("strong_title")
    return tuple(tags)


def _selection_rank(offer_id: str, *, seed: str) -> str:
    digest = hashlib.sha256(f"{seed}:{offer_id}".encode("utf-8")).hexdigest()
    return digest


def select_arena_sample_records(
    candidates: tuple[ArenaSampleRecord, ...],
    *,
    target_size: int = ARENA_SAMPLE_TARGET_SIZE,
    seed: str = ARENA_SAMPLE_SELECTION_SEED,
    strata_quotas: tuple[tuple[str, int], ...] = STRATA_QUOTAS,
) -> tuple[ArenaSampleRecord, ...]:
    if target_size <= 0:
        msg = "target_size must be > 0"
        raise ValueError(msg)

    selected: dict[str, ArenaSampleRecord] = {}
    quota_remaining = {tag: quota for tag, quota in strata_quotas}

    ordered = sorted(candidates, key=lambda record: _selection_rank(record.offer_id, seed=seed))
    for record in ordered:
        if len(selected) >= target_size:
            break
        if record.offer_id in selected:
            continue
        for tag in record.strata_tags:
            remaining = quota_remaining.get(tag)
            if remaining is not None and remaining > 0:
                selected[record.offer_id] = record
                for applied_tag in record.strata_tags:
                    if applied_tag in quota_remaining and quota_remaining[applied_tag] > 0:
                        quota_remaining[applied_tag] -= 1
                break

    if len(selected) < target_size:
        for record in ordered:
            if len(selected) >= target_size:
                break
            if record.offer_id not in selected:
                selected[record.offer_id] = record

    final_records = tuple(
        selected[offer_id]
        for offer_id in sorted(
            selected,
            key=lambda offer_id: _selection_rank(offer_id, seed=seed),
        )
    )
    return final_records[:target_size]


def build_arena_sample_manifest(
    records: tuple[ArenaSampleRecord, ...],
    *,
    scan_row_limit: int = ARENA_SAMPLE_SCAN_ROW_LIMIT,
    seed: str = ARENA_SAMPLE_SELECTION_SEED,
    strata_quotas: tuple[tuple[str, int], ...] = STRATA_QUOTAS,
) -> ArenaSampleManifest:
    snapshots = tuple(
        ArenaSampleRecordSnapshot(
            offer_id=record.offer_id,
            global_row_index=record.global_row_index,
            semantic_text=record.semantic_text,
            strata_tags=record.strata_tags,
            benchmark_only_cluster_id=record.source_offer.cluster_id,
        )
        for record in records
    )
    return ArenaSampleManifest(
        version=ARENA_SAMPLE_VERSION,
        selection_seed=seed,
        scan_row_limit=scan_row_limit,
        target_size=len(records),
        strata_quotas=strata_quotas,
        records=snapshots,
    )
