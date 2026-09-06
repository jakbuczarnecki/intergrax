"""Deterministic stratified sample selection for proof-50."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import pyarrow.parquet as pq

from platform_proofs.scenarios.verified_product_identification.application.domain.wdc_source_offer import (
    parse_wdc_source_offer_json,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.errors import (
    VpiDataPackBuildError,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.identity import (
    PROOF_50_RECORD_COUNT,
    PROOF_50_SAMPLE_VERSION,
)

PROOF_50_SAMPLE_SEED = 42


@dataclass(frozen=True, slots=True)
class SelectedDatasetRow:
    global_row_index: int
    record_json: str
    offer_id: str


@dataclass(frozen=True, slots=True)
class DiversitySignature:
    has_identifiers: bool
    has_brand: bool
    has_category: bool
    long_description: bool
    has_spec_table: bool
    has_structured_attributes: bool

    def key(self) -> str:
        return (
            f"ident={int(self.has_identifiers)}"
            f"|brand={int(self.has_brand)}"
            f"|category={int(self.has_category)}"
            f"|long_desc={int(self.long_description)}"
            f"|spec={int(self.has_spec_table)}"
            f"|struct={int(self.has_structured_attributes)}"
        )


def _rank_key(offer_id: str) -> str:
    payload = f"{PROOF_50_SAMPLE_VERSION}|{PROOF_50_SAMPLE_SEED}|{offer_id}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _diversity_signature(record_json: str) -> DiversitySignature:
    source_offer = parse_wdc_source_offer_json(record_json)
    description = source_offer.description or ""
    return DiversitySignature(
        has_identifiers=len(source_offer.identifiers) > 0,
        has_brand=source_offer.brand is not None and bool(source_offer.brand.strip()),
        has_category=source_offer.category is not None and bool(source_offer.category.strip()),
        long_description=len(description) >= 120,
        has_spec_table=source_offer.spec_table_content is not None
        and bool(source_offer.spec_table_content.strip()),
        has_structured_attributes=len(source_offer.key_value_pairs) > 0,
    )


def select_proof_sample_rows(
    dataset_path: str,
    *,
    record_count: int = PROOF_50_RECORD_COUNT,
    batch_size: int = 10_000,
) -> tuple[SelectedDatasetRow, ...]:
    parquet_file = pq.ParquetFile(dataset_path)
    stratum_best: dict[str, tuple[str, SelectedDatasetRow]] = {}
    global_best: list[tuple[str, SelectedDatasetRow]] = []

    global_row_index = 0
    for batch in parquet_file.iter_batches(columns=["record_json"], batch_size=batch_size):
        column = batch.column(0)
        for index in range(batch.num_rows):
            record_json = column[index].as_py()
            if not isinstance(record_json, str):
                msg = "record_json column must contain UTF-8 strings"
                raise TypeError(msg)
            source_offer = parse_wdc_source_offer_json(record_json)
            row = SelectedDatasetRow(
                global_row_index=global_row_index,
                record_json=record_json,
                offer_id=source_offer.offer_id,
            )
            rank = _rank_key(source_offer.offer_id)
            signature = _diversity_signature(record_json)
            stratum_key = signature.key()
            existing = stratum_best.get(stratum_key)
            if existing is None or rank < existing[0]:
                stratum_best[stratum_key] = (rank, row)
            global_best.append((rank, row))
            global_row_index += 1

    selected: dict[str, SelectedDatasetRow] = {}
    for rank, row in sorted(stratum_best.values(), key=lambda item: item[0]):
        selected[row.offer_id] = row
        if len(selected) >= record_count:
            break

    if len(selected) < record_count:
        for rank, row in sorted(global_best, key=lambda item: item[0]):
            if row.offer_id in selected:
                continue
            selected[row.offer_id] = row
            if len(selected) >= record_count:
                break

    if len(selected) < record_count:
        raise VpiDataPackBuildError(
            f"dataset yielded only {len(selected)} selectable rows; requested {record_count}"
        )

    ordered = tuple(
        sorted(selected.values(), key=lambda row: _rank_key(row.offer_id))[:record_count]
    )
    if len(ordered) != record_count:
        raise VpiDataPackBuildError(
            f"selection produced {len(ordered)} rows; expected {record_count}"
        )
    return ordered
