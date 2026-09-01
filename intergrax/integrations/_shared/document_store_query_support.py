# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Shared helpers for bounded DocumentStore data-path queries."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from intergrax.integrations.contracts.document_store import (
    DocumentDataEquality,
    DocumentDataSort,
    DocumentQueryCursorCodec,
    DocumentQueryCursorPayloadV2,
    DocumentRecord,
    _ROW_KEY_SORT_PATH,
    document_data_path_value,
    document_sort_key_values,
    normalize_document_data_equalities,
    normalize_document_data_sort,
)


def _canonical_json_value(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _decode_canonical_json_value(value_json: str) -> Any:
    return json.loads(value_json)


def document_matches_equalities(
    document: DocumentRecord,
    equalities: Sequence[DocumentDataEquality],
) -> bool:
    for equality in equalities:
        actual = document_data_path_value(document.data, equality.path)
        if actual != equality.value:
            return False
    return True


def sort_documents(
    documents: Sequence[DocumentRecord],
    sort: Sequence[DocumentDataSort],
) -> list[DocumentRecord]:
    rows = list(documents)
    if not sort:
        rows.sort(key=lambda doc: doc.row_key)
        return rows
    rows.sort(
        key=lambda doc: document_sort_key_values(doc, sort),
        reverse=_sort_is_descending(sort),
    )
    return rows


def _sort_is_descending(sort: Sequence[DocumentDataSort]) -> bool:
    directions = {spec.direction for spec in sort}
    if len(directions) != 1:
        raise ValueError("document_store_data_sort_invalid")
    return directions.pop() == "desc"


def document_before_sort_cursor(
    document: DocumentRecord,
    sort: Sequence[DocumentDataSort],
    cursor_values: tuple[Any, ...],
) -> bool:
    return document_sort_key_values(document, sort) < cursor_values


def decode_v2_sort_values(
    payload: DocumentQueryCursorPayloadV2,
) -> tuple[Any, ...]:
    return tuple(_decode_canonical_json_value(value_json) for value_json in payload.last_sort_values)


def build_mongo_data_field(path: str) -> str:
    if path == _ROW_KEY_SORT_PATH:
        return "row_key"
    return f"data.{path}"


def build_mongo_equality_filter(
    equalities: Sequence[DocumentDataEquality],
) -> dict[str, Any]:
    mongo_filter: dict[str, Any] = {}
    for equality in equalities:
        mongo_filter[build_mongo_data_field(equality.path)] = equality.value
    return mongo_filter


def build_mongo_keyset_filter(
    sort: Sequence[DocumentDataSort],
    sort_values: tuple[Any, ...],
) -> dict[str, Any]:
    if not sort:
        raise ValueError("document_store_data_sort_invalid")
    clauses: list[dict[str, Any]] = []
    for index, spec in enumerate(sort):
        clause: dict[str, Any] = {}
        for prior_index in range(index):
            prior_spec = sort[prior_index]
            clause[build_mongo_data_field(prior_spec.path)] = sort_values[prior_index]
        field = build_mongo_data_field(spec.path)
        value = sort_values[index]
        if spec.direction == "desc":
            clause[field] = {"$lt": value}
        else:
            clause[field] = {"$gt": value}
        clauses.append(clause)
    return {"$or": clauses}


def query_documents_with_data_filters(
    *,
    rows: Sequence[DocumentRecord],
    partition_key: str,
    limit: int,
    row_key_prefix: str | None,
    row_key_upper_bound: str | None,
    data_equalities: Sequence[DocumentDataEquality],
    sort: Sequence[DocumentDataSort],
    cursor_codec: DocumentQueryCursorCodec,
    cursor: str | None,
    rows_examined_counter: list[int] | None = None,
) -> tuple[tuple[DocumentRecord, ...], str | None]:
    normalized_equalities = normalize_document_data_equalities(data_equalities)
    normalized_sort = normalize_document_data_sort(sort)
    bounded_limit = limit

    candidates: list[DocumentRecord] = []
    for document in rows:
        if document.partition_key != partition_key:
            continue
        if row_key_prefix is not None and not document.row_key.startswith(row_key_prefix):
            continue
        if row_key_upper_bound is not None and document.row_key > row_key_upper_bound:
            continue
        if rows_examined_counter is not None:
            rows_examined_counter[0] += 1
        if normalized_equalities and not document_matches_equalities(document, normalized_equalities):
            continue
        candidates.append(document)

    if normalized_sort:
        candidates = sort_documents(candidates, normalized_sort)
        if cursor is not None:
            payload = cursor_codec.decode_v2(
                cursor,
                partition_key=partition_key,
                row_key_prefix=row_key_prefix,
                row_key_upper_bound=row_key_upper_bound,
                data_equalities=normalized_equalities,
                sort=normalized_sort,
            )
            cursor_values = decode_v2_sort_values(payload)
            candidates = [
                document
                for document in candidates
                if document_before_sort_cursor(document, normalized_sort, cursor_values)
            ]
        page = candidates[:bounded_limit]
        has_more = len(candidates) > bounded_limit
        next_cursor = (
            cursor_codec.encode_v2(
                partition_key=partition_key,
                row_key_prefix=row_key_prefix,
                row_key_upper_bound=row_key_upper_bound,
                data_equalities=normalized_equalities,
                sort=normalized_sort,
                last_row_key=page[-1].row_key,
                last_sort_values=document_sort_key_values(page[-1], normalized_sort),
            )
            if has_more and page
            else None
        )
        return tuple(page), next_cursor

    candidates.sort(key=lambda doc: doc.row_key)
    if cursor is not None:
        last_row_key = cursor_codec.decode(
            cursor,
            partition_key=partition_key,
            row_key_prefix=row_key_prefix,
        ).last_row_key
        candidates = [document for document in candidates if document.row_key > last_row_key]
    page = candidates[:bounded_limit]
    next_cursor = (
        cursor_codec.encode(
            partition_key=partition_key,
            row_key_prefix=row_key_prefix,
            last_row_key=page[-1].row_key,
        )
        if len(candidates) > bounded_limit and page
        else None
    )
    return tuple(page), next_cursor
