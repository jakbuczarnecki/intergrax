"""Offline statistical profiler for selected_offers.parquet."""

from __future__ import annotations

import argparse
import json
import sys
import time
import tracemalloc
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeAlias

import pyarrow.parquet as pq

PROFILER_VERSION = "verified_product_identification_wdc_profiler/1.0.0"
PROFILE_VERSION = "1.0.0"
DATASET_DIR = Path(__file__).resolve().parent
DEFAULT_BATCH_SIZE = 10_000
PROGRESS_INTERVAL_RECORDS = 250_000

STRING_FIELDS = ("title", "description", "brand", "price", "specTableContent")
GTIN_KEY_MARKERS = ("/gtin8", "/gtin12", "/gtin13", "/gtin14", "gtin")

JsonPrimitive: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonPrimitive | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject: TypeAlias = dict[str, JsonValue]

ATTRIBUTE_COUNT_BUCKETS: tuple[tuple[str, int, int | None], ...] = (
    ("0", 0, 0),
    ("1", 1, 1),
    ("2-5", 2, 5),
    ("6-10", 6, 10),
    ("11-20", 11, 20),
    ("21-50", 21, 50),
    ("51-100", 51, 100),
    (">100", 101, None),
)

CLUSTER_SIZE_BUCKETS: tuple[tuple[str, int, int | None], ...] = (
    ("1", 1, 1),
    ("2", 2, 2),
    ("3", 3, 3),
    ("4", 4, 4),
    ("5", 5, 5),
    ("6-10", 6, 10),
    ("11-20", 11, 20),
    ("21-50", 21, 50),
    ("51-100", 51, 100),
    (">100", 101, None),
)


def _bucket_label(
    value: int,
    buckets: tuple[tuple[str, int, int | None], ...],
) -> str:
    for label, lower, upper in buckets:
        if upper is None:
            if value >= lower:
                return label
        elif lower <= value <= upper:
            return label
    return buckets[-1][0]


class LengthHistogram:
    """Bounded-memory length histogram for approximate percentile stats."""

    METHOD = "fixed-width histogram with exact buckets 0-200, then widening bins"

    def __init__(self) -> None:
        self.count = 0
        self.sum_lengths = 0
        self.min_length: int | None = None
        self.max_length: int | None = None
        self._buckets: Counter[int] = Counter()

    def add(self, length: int) -> None:
        self.count += 1
        self.sum_lengths += length
        self.min_length = length if self.min_length is None else min(self.min_length, length)
        self.max_length = length if self.max_length is None else max(self.max_length, length)
        self._buckets[self._bucket_upper_bound(length)] += 1

    @staticmethod
    def _bucket_upper_bound(length: int) -> int:
        if length <= 200:
            return length
        if length <= 500:
            return 500
        if length <= 1000:
            return 1000
        if length <= 2000:
            return 2000
        if length <= 5000:
            return 5000
        if length <= 10000:
            return 10000
        if length <= 20000:
            return 20000
        if length <= 50000:
            return 50000
        return 100_000

    def _percentile(self, percentile: float) -> int | None:
        if self.count == 0:
            return None
        target = max(1, int(self.count * percentile))
        cumulative = 0
        for upper in sorted(self._buckets):
            cumulative += self._buckets[upper]
            if cumulative >= target:
                return upper
        return self.max_length

    def summary(self) -> dict[str, Any]:
        if self.count == 0:
            return {
                "count": 0,
                "min_length": None,
                "max_length": None,
                "average_length": None,
                "median_length": None,
                "p90_length": None,
                "p95_length": None,
                "p99_length": None,
                "percentile_method": self.METHOD,
                "percentiles_approximate": True,
            }
        return {
            "count": self.count,
            "min_length": self.min_length,
            "max_length": self.max_length,
            "average_length": round(self.sum_lengths / self.count, 2),
            "median_length": self._percentile(0.5),
            "p90_length": self._percentile(0.90),
            "p95_length": self._percentile(0.95),
            "p99_length": self._percentile(0.99),
            "percentile_method": self.METHOD,
            "percentiles_approximate": True,
        }


@dataclass
class StringFieldStats:
    missing: int = 0
    null: int = 0
    empty: int = 0
    non_empty: int = 0
    length_histogram: LengthHistogram = field(default_factory=LengthHistogram)

    def observe(self, record: JsonObject, field_name: str) -> None:
        if field_name not in record:
            self.missing += 1
            return
        value = record[field_name]
        if value is None:
            self.null += 1
            return
        if not isinstance(value, str):
            return
        if not value.strip():
            self.empty += 1
            return
        self.non_empty += 1
        self.length_histogram.add(len(value))

    def to_dict(self) -> dict[str, Any]:
        result = {
            "missing": self.missing,
            "null": self.null,
            "empty": self.empty,
            "non_empty": self.non_empty,
        }
        result.update(self.length_histogram.summary())
        return result


@dataclass
class CategoryAccumulator:
    record_count: int = 0
    with_key_value_pairs: int = 0
    with_spec_table_content: int = 0
    with_brand: int = 0
    with_description: int = 0
    with_price: int = 0
    with_identifiers: int = 0
    kvp_count_sum: int = 0


@dataclass
class ProfileAccumulator:
    total_records: int = 0
    malformed_record_json_count: int = 0
    non_object_record_count: int = 0
    unexpected_field_type_counts: Counter[str] = field(default_factory=Counter)

    top_level_field_present: Counter[str] = field(default_factory=Counter)
    top_level_field_null: Counter[str] = field(default_factory=Counter)
    top_level_field_non_null: Counter[str] = field(default_factory=Counter)

    missing_category_count: int = 0
    null_category_count: int = 0
    empty_category_count: int = 0
    categories: dict[str, CategoryAccumulator] = field(default_factory=dict)

    records_with_identifiers: int = 0
    records_without_identifiers: int = 0
    total_identifier_entries: int = 0
    max_identifiers_per_record: int = 0
    identifier_key_record_counts: Counter[str] = field(default_factory=Counter)
    identifier_key_occurrence_counts: Counter[str] = field(default_factory=Counter)
    records_with_any_gtin: int = 0
    records_with_mpn: int = 0
    records_with_sku: int = 0
    records_with_product_id: int = 0
    records_with_multiple_identifier_types: int = 0
    empty_identifier_value_count: int = 0
    duplicate_identifier_entry_count: int = 0

    records_with_cluster_id: int = 0
    records_without_cluster_id: int = 0
    cluster_counts: Counter[int] = field(default_factory=Counter)

    records_with_key_value_pairs: int = 0
    records_without_key_value_pairs: int = 0
    total_attribute_entries: int = 0
    min_attribute_count: int | None = None
    max_attribute_count: int = 0
    attribute_count_distribution: Counter[str] = field(default_factory=Counter)
    attribute_name_occurrence_counts: Counter[str] = field(default_factory=Counter)
    attribute_name_record_counts: Counter[str] = field(default_factory=Counter)

    string_fields: dict[str, StringFieldStats] = field(
        default_factory=lambda: {name: StringFieldStats() for name in STRING_FIELDS}
    )

    records_with_empty_title: int = 0
    records_without_brand: int = 0
    records_without_description: int = 0
    records_without_price: int = 0
    records_without_category: int = 0
    records_with_spec_but_no_kvp: int = 0
    records_with_kvp_but_no_spec: int = 0
    records_with_both_spec_and_kvp: int = 0
    records_with_very_long_title: int = 0
    records_with_very_long_description: int = 0
    records_with_very_long_spec: int = 0


def iter_record_json(input_path: Path, *, batch_size: int) -> Iterator[str]:
    parquet_file = pq.ParquetFile(input_path)
    for batch in parquet_file.iter_batches(columns=["record_json"], batch_size=batch_size):
        column = batch.column(0)
        for index in range(batch.num_rows):
            value = column[index].as_py()
            if not isinstance(value, str):
                msg = "record_json column must contain UTF-8 strings"
                raise TypeError(msg)
            yield value


def _note_unexpected_type(
    accumulator: ProfileAccumulator,
    field_name: str,
    expected: str,
    actual: object,
) -> None:
    actual_type = type(actual).__name__
    key = f"{field_name}:expected_{expected}_got_{actual_type}"
    accumulator.unexpected_field_type_counts[key] += 1


def _is_non_empty_string(value: JsonValue) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _category_key(value: JsonValue, accumulator: ProfileAccumulator) -> str | None:
    if value is None:
        accumulator.null_category_count += 1
        return None
    if not isinstance(value, str):
        _note_unexpected_type(accumulator, "category", "str_or_null", value)
        return None
    if not value.strip():
        accumulator.empty_category_count += 1
        return None
    return value


def _get_category_accumulator(
    accumulator: ProfileAccumulator,
    category: str | None,
) -> CategoryAccumulator | None:
    if category is None:
        return None
    if category not in accumulator.categories:
        accumulator.categories[category] = CategoryAccumulator()
    return accumulator.categories[category]


def _process_identifiers(
    record: JsonObject,
    accumulator: ProfileAccumulator,
    category_acc: CategoryAccumulator | None,
) -> None:
    if "identifiers" not in record:
        accumulator.records_without_identifiers += 1
        return

    identifiers = record["identifiers"]
    if identifiers is None:
        accumulator.records_without_identifiers += 1
        return
    if not isinstance(identifiers, list):
        _note_unexpected_type(accumulator, "identifiers", "list_or_null", identifiers)
        return

    if not identifiers:
        accumulator.records_without_identifiers += 1
        return

    accumulator.records_with_identifiers += 1
    if category_acc is not None:
        category_acc.with_identifiers += 1

    seen_keys: set[str] = set()
    identifier_types_in_record: set[str] = set()
    entry_count = 0

    for entry in identifiers:
        if not isinstance(entry, dict):
            _note_unexpected_type(accumulator, "identifiers[]", "object", entry)
            continue
        if len(entry) != 1:
            accumulator.duplicate_identifier_entry_count += 1
        for key, value in entry.items():
            entry_count += 1
            identifier_types_in_record.add(key)
            accumulator.identifier_key_occurrence_counts[key] += 1
            if key in seen_keys:
                accumulator.duplicate_identifier_entry_count += 1
            seen_keys.add(key)
            if value is None or (isinstance(value, str) and not value.strip()):
                accumulator.empty_identifier_value_count += 1

    for key in identifier_types_in_record:
        accumulator.identifier_key_record_counts[key] += 1

    accumulator.total_identifier_entries += entry_count
    accumulator.max_identifiers_per_record = max(
        accumulator.max_identifiers_per_record,
        entry_count,
    )

    if any(
        any(marker in key.lower() for marker in GTIN_KEY_MARKERS)
        for key in identifier_types_in_record
    ):
        accumulator.records_with_any_gtin += 1
    if any("mpn" in key.lower() for key in identifier_types_in_record):
        accumulator.records_with_mpn += 1
    if any("sku" in key.lower() for key in identifier_types_in_record):
        accumulator.records_with_sku += 1
    if any("productid" in key.lower() for key in identifier_types_in_record):
        accumulator.records_with_product_id += 1
    if len(identifier_types_in_record) > 1:
        accumulator.records_with_multiple_identifier_types += 1


def _process_key_value_pairs(
    record: JsonObject,
    accumulator: ProfileAccumulator,
    category_acc: CategoryAccumulator | None,
) -> tuple[bool, int]:
    if "keyValuePairs" not in record:
        accumulator.records_without_key_value_pairs += 1
        accumulator.attribute_count_distribution["0"] += 1
        return False, 0

    kvp = record["keyValuePairs"]
    if kvp is None:
        accumulator.records_without_key_value_pairs += 1
        accumulator.attribute_count_distribution["0"] += 1
        return False, 0
    if not isinstance(kvp, dict):
        _note_unexpected_type(accumulator, "keyValuePairs", "dict_or_null", kvp)
        accumulator.records_without_key_value_pairs += 1
        accumulator.attribute_count_distribution["0"] += 1
        return False, 0

    attribute_count = len(kvp)
    if attribute_count == 0:
        accumulator.records_without_key_value_pairs += 1
        accumulator.attribute_count_distribution["0"] += 1
        return False, 0

    accumulator.records_with_key_value_pairs += 1
    if category_acc is not None:
        category_acc.with_key_value_pairs += 1
        category_acc.kvp_count_sum += attribute_count

    accumulator.total_attribute_entries += attribute_count
    if accumulator.min_attribute_count is None:
        accumulator.min_attribute_count = attribute_count
    else:
        accumulator.min_attribute_count = min(accumulator.min_attribute_count, attribute_count)
    accumulator.max_attribute_count = max(accumulator.max_attribute_count, attribute_count)
    bucket = _bucket_label(attribute_count, ATTRIBUTE_COUNT_BUCKETS)
    accumulator.attribute_count_distribution[bucket] += 1

    for attribute_name in kvp:
        accumulator.attribute_name_occurrence_counts[attribute_name] += 1
        accumulator.attribute_name_record_counts[attribute_name] += 1

    return True, attribute_count


def _process_spec_table_content(
    record: JsonObject,
    category_acc: CategoryAccumulator | None,
) -> bool:
    if "specTableContent" not in record:
        return False
    value = record["specTableContent"]
    if value is None:
        return False
    if not isinstance(value, str):
        return False
    if not value.strip():
        return False
    if category_acc is not None:
        category_acc.with_spec_table_content += 1
    return True


def _process_cluster_id(
    record: JsonObject,
    accumulator: ProfileAccumulator,
) -> None:
    if "cluster_id" not in record:
        accumulator.records_without_cluster_id += 1
        return
    cluster_id = record["cluster_id"]
    if cluster_id is None:
        accumulator.records_without_cluster_id += 1
        return
    if not isinstance(cluster_id, int):
        _note_unexpected_type(accumulator, "cluster_id", "int_or_null", cluster_id)
        accumulator.records_without_cluster_id += 1
        return
    accumulator.records_with_cluster_id += 1
    accumulator.cluster_counts[cluster_id] += 1


def _process_top_level_fields(accumulator: ProfileAccumulator, record: JsonObject) -> None:
    for field_name, value in record.items():
        accumulator.top_level_field_present[field_name] += 1
        if value is None:
            accumulator.top_level_field_null[field_name] += 1
        else:
            accumulator.top_level_field_non_null[field_name] += 1


def _process_record(accumulator: ProfileAccumulator, record: JsonObject) -> None:
    accumulator.total_records += 1
    _process_top_level_fields(accumulator, record)

    if "category" not in record:
        accumulator.missing_category_count += 1
        accumulator.records_without_category += 1
        category = None
    else:
        category = _category_key(record["category"], accumulator)
        if category is None:
            accumulator.records_without_category += 1

    category_acc = _get_category_accumulator(accumulator, category)

    has_kvp, _ = _process_key_value_pairs(record, accumulator, category_acc)
    has_spec = _process_spec_table_content(record, category_acc)

    if has_spec and not has_kvp:
        accumulator.records_with_spec_but_no_kvp += 1
    if has_kvp and not has_spec:
        accumulator.records_with_kvp_but_no_spec += 1
    if has_spec and has_kvp:
        accumulator.records_with_both_spec_and_kvp += 1

    if _is_non_empty_string(record.get("brand")):
        if category_acc is not None:
            category_acc.with_brand += 1
    else:
        accumulator.records_without_brand += 1

    if _is_non_empty_string(record.get("description")):
        if category_acc is not None:
            category_acc.with_description += 1
    else:
        accumulator.records_without_description += 1

    if _is_non_empty_string(record.get("price")):
        if category_acc is not None:
            category_acc.with_price += 1
    else:
        accumulator.records_without_price += 1

    title_value = record.get("title")
    if not _is_non_empty_string(title_value):
        accumulator.records_with_empty_title += 1
    elif isinstance(title_value, str) and len(title_value) > 500:
        accumulator.records_with_very_long_title += 1

    description_value = record.get("description")
    if isinstance(description_value, str) and len(description_value) > 5000:
        accumulator.records_with_very_long_description += 1

    spec_value = record.get("specTableContent")
    if isinstance(spec_value, str) and len(spec_value) > 5000:
        accumulator.records_with_very_long_spec += 1

    for field_name, stats in accumulator.string_fields.items():
        stats.observe(record, field_name)

    _process_identifiers(record, accumulator, category_acc)
    _process_cluster_id(record, accumulator)

    if category_acc is not None:
        category_acc.record_count += 1


def profile_dataset(
    *,
    input_path: Path,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> tuple[ProfileAccumulator, int | None]:
    if not input_path.is_file():
        msg = f"input file does not exist: {input_path}"
        raise FileNotFoundError(msg)
    if batch_size <= 0:
        msg = "batch_size must be positive"
        raise ValueError(msg)

    accumulator = ProfileAccumulator()
    tracemalloc.start()

    for index, raw_json in enumerate(iter_record_json(input_path, batch_size=batch_size), start=1):
        try:
            parsed = json.loads(raw_json)
        except json.JSONDecodeError:
            accumulator.malformed_record_json_count += 1
            continue

        if not isinstance(parsed, dict):
            accumulator.non_object_record_count += 1
            continue

        _process_record(accumulator, parsed)

        if index % PROGRESS_INTERVAL_RECORDS == 0:
            print(
                f"[progress] processed={index:,} total={accumulator.total_records:,}",
                file=sys.stderr,
            )

    _, peak_memory_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return accumulator, peak_memory_bytes


def _percent(part: int, whole: int) -> float:
    if whole == 0:
        return 0.0
    return round(100.0 * part / whole, 4)


def _sorted_counter_items(counter: Counter[str]) -> list[tuple[str, int]]:
    return sorted(counter.items(), key=lambda item: (-item[1], item[0]))


def _build_top_level_fields(
    accumulator: ProfileAccumulator,
    total_records: int,
) -> dict[str, Any]:
    all_fields = set(accumulator.top_level_field_present) | set(
        accumulator.top_level_field_null
    ) | set(accumulator.top_level_field_non_null)
    fields: dict[str, Any] = {}
    for field_name in sorted(all_fields):
        present_count = accumulator.top_level_field_present[field_name]
        null_count = accumulator.top_level_field_null[field_name]
        non_null_count = accumulator.top_level_field_non_null[field_name]
        missing_count = total_records - present_count
        fields[field_name] = {
            "present_count": present_count,
            "null_count": null_count,
            "non_null_count": non_null_count,
            "missing_count": missing_count,
            "non_null_percent": _percent(non_null_count, total_records),
        }
    return fields


def _build_categories(accumulator: ProfileAccumulator, total_records: int) -> dict[str, Any]:
    category_rows: list[dict[str, Any]] = []
    for category_name, stats in sorted(
        accumulator.categories.items(),
        key=lambda item: (-item[1].record_count, item[0]),
    ):
        record_count = stats.record_count
        category_rows.append(
            {
                "category": category_name,
                "record_count": record_count,
                "percent_of_dataset": _percent(record_count, total_records),
                "with_key_value_pairs": stats.with_key_value_pairs,
                "with_spec_table_content": stats.with_spec_table_content,
                "with_brand": stats.with_brand,
                "with_description": stats.with_description,
                "with_price": stats.with_price,
                "with_identifiers": stats.with_identifiers,
                "avg_kvp_count": round(stats.kvp_count_sum / record_count, 4)
                if record_count
                else 0.0,
                "identifier_coverage_percent": _percent(stats.with_identifiers, record_count),
                "brand_coverage_percent": _percent(stats.with_brand, record_count),
                "description_coverage_percent": _percent(stats.with_description, record_count),
                "price_coverage_percent": _percent(stats.with_price, record_count),
                "multi_offer_cluster_coverage_percent": None,
                "multi_offer_cluster_coverage_skipped_reason": (
                    "Per-category multi-offer cluster coverage requires a second pass "
                    "or per-cluster category mapping; skipped in single-pass profiler."
                ),
            }
        )

    return {
        "unique_category_count": len(accumulator.categories),
        "missing_category_count": accumulator.missing_category_count,
        "null_category_count": accumulator.null_category_count,
        "empty_category_count": accumulator.empty_category_count,
        "items": category_rows,
        "top_categories": category_rows[:20],
    }


def _build_identifiers(accumulator: ProfileAccumulator, total_records: int) -> dict[str, Any]:
    by_key = [
        {
            "identifier_key": key,
            "record_count": count,
            "total_occurrences": accumulator.identifier_key_occurrence_counts[key],
            "percent_of_dataset": _percent(count, total_records),
        }
        for key, count in _sorted_counter_items(accumulator.identifier_key_record_counts)
    ]
    records_with_kvp = accumulator.records_with_key_value_pairs
    return {
        "records_with_identifiers": accumulator.records_with_identifiers,
        "records_without_identifiers": accumulator.records_without_identifiers,
        "total_identifier_entries": accumulator.total_identifier_entries,
        "average_identifiers_per_record": round(
            accumulator.total_identifier_entries / total_records,
            4,
        )
        if total_records
        else 0.0,
        "max_identifiers_per_record": accumulator.max_identifiers_per_record,
        "by_key": by_key,
        "records_with_any_gtin": accumulator.records_with_any_gtin,
        "records_with_mpn": accumulator.records_with_mpn,
        "records_with_sku": accumulator.records_with_sku,
        "records_with_product_id": accumulator.records_with_product_id,
        "records_with_multiple_identifier_types": (
            accumulator.records_with_multiple_identifier_types
        ),
        "empty_identifier_value_count": accumulator.empty_identifier_value_count,
        "duplicate_identifier_entry_count": accumulator.duplicate_identifier_entry_count,
    }


def _build_clusters(accumulator: ProfileAccumulator) -> dict[str, Any]:
    cluster_sizes = list(accumulator.cluster_counts.values())
    unique_cluster_count = len(accumulator.cluster_counts)
    singleton_cluster_count = sum(1 for size in cluster_sizes if size == 1)
    multi_offer_cluster_count = sum(1 for size in cluster_sizes if size > 1)
    records_in_multi_offer_clusters = sum(size for size in cluster_sizes if size > 1)

    size_distribution: Counter[str] = Counter()
    for size in cluster_sizes:
        size_distribution[_bucket_label(size, CLUSTER_SIZE_BUCKETS)] += 1

    size_frequency = Counter(cluster_sizes)
    median_cluster_size = None
    if cluster_sizes:
        cumulative = 0
        target = (unique_cluster_count + 1) // 2
        for size in sorted(size_frequency):
            cumulative += size_frequency[size]
            if cumulative >= target:
                median_cluster_size = size
                break

    top_clusters = [
        {"cluster_id": cluster_id, "record_count": count}
        for cluster_id, count in accumulator.cluster_counts.most_common(50)
    ]

    max_cluster_size = max(cluster_sizes) if cluster_sizes else 0
    average_cluster_size = (
        round(sum(cluster_sizes) / unique_cluster_count, 4) if unique_cluster_count else 0.0
    )

    return {
        "records_with_cluster_id": accumulator.records_with_cluster_id,
        "records_without_cluster_id": accumulator.records_without_cluster_id,
        "unique_cluster_count": unique_cluster_count,
        "singleton_cluster_count": singleton_cluster_count,
        "multi_offer_cluster_count": multi_offer_cluster_count,
        "records_in_multi_offer_clusters": records_in_multi_offer_clusters,
        "size_distribution": {
            label: size_distribution.get(label, 0)
            for label, _, _ in CLUSTER_SIZE_BUCKETS
        },
        "max_cluster_size": max_cluster_size,
        "average_cluster_size": average_cluster_size,
        "median_cluster_size": median_cluster_size,
        "median_cluster_size_method": "exact from cluster-size frequency table",
        "top_clusters": top_clusters,
    }


def _build_key_value_pairs(accumulator: ProfileAccumulator) -> dict[str, Any]:
    attribute_items = [
        {
            "attribute_name": name,
            "occurrence_count": accumulator.attribute_name_occurrence_counts[name],
            "record_count": accumulator.attribute_name_record_counts[name],
        }
        for name, _ in _sorted_counter_items(accumulator.attribute_name_occurrence_counts)
    ]
    records_with_kvp = accumulator.records_with_key_value_pairs
    return {
        "records_with_key_value_pairs": records_with_kvp,
        "records_without_key_value_pairs": accumulator.records_without_key_value_pairs,
        "total_attribute_entries": accumulator.total_attribute_entries,
        "average_attribute_count_per_record_with_kvp": round(
            accumulator.total_attribute_entries / records_with_kvp,
            4,
        )
        if records_with_kvp
        else 0.0,
        "min_attribute_count": accumulator.min_attribute_count,
        "max_attribute_count": accumulator.max_attribute_count,
        "attribute_count_distribution": {
            label: accumulator.attribute_count_distribution.get(label, 0)
            for label, _, _ in ATTRIBUTE_COUNT_BUCKETS
        },
        "unique_attribute_name_count": len(accumulator.attribute_name_occurrence_counts),
        "attribute_names": attribute_items,
        "top_attribute_names": attribute_items[:200],
    }


def build_profile_document(
    *,
    input_path: Path,
    accumulator: ProfileAccumulator,
    profiling_started_at: datetime,
    profiling_completed_at: datetime,
    peak_memory_bytes: int | None,
) -> dict[str, Any]:
    total_records = accumulator.total_records
    duration_seconds = round(
        (profiling_completed_at - profiling_started_at).total_seconds(),
        3,
    )

    return {
        "profile_version": PROFILE_VERSION,
        "dataset": {
            "source_file": str(input_path.resolve()),
            "total_records": total_records,
            "profiling_started_at": profiling_started_at.isoformat(),
            "profiling_completed_at": profiling_completed_at.isoformat(),
            "profiling_duration_seconds": duration_seconds,
            "profiler_version": PROFILER_VERSION,
            "peak_memory_bytes": peak_memory_bytes,
        },
        "top_level_fields": _build_top_level_fields(accumulator, total_records),
        "categories": _build_categories(accumulator, total_records),
        "identifiers": _build_identifiers(accumulator, total_records),
        "clusters": _build_clusters(accumulator),
        "key_value_pairs": _build_key_value_pairs(accumulator),
        "string_fields": {
            field_name: stats.to_dict()
            for field_name, stats in accumulator.string_fields.items()
        },
        "quality": {
            "records_with_empty_title": accumulator.records_with_empty_title,
            "records_without_brand": accumulator.records_without_brand,
            "records_without_description": accumulator.records_without_description,
            "records_without_price": accumulator.records_without_price,
            "records_without_category": accumulator.records_without_category,
            "records_with_spec_but_no_kvp": accumulator.records_with_spec_but_no_kvp,
            "records_with_kvp_but_no_spec": accumulator.records_with_kvp_but_no_spec,
            "records_with_both_spec_and_kvp": accumulator.records_with_both_spec_and_kvp,
            "records_with_very_long_title": accumulator.records_with_very_long_title,
            "records_with_very_long_description": accumulator.records_with_very_long_description,
            "records_with_very_long_spec": accumulator.records_with_very_long_spec,
        },
        "contract_violations": {
            "malformed_record_json_count": accumulator.malformed_record_json_count,
            "non_object_record_count": accumulator.non_object_record_count,
            "unexpected_field_type_counts": dict(
                sorted(accumulator.unexpected_field_type_counts.items())
            ),
        },
    }


def write_profile_json(output_path: Path, profile: dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(profile, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_profile_markdown(output_path: Path, profile: dict[str, Any]) -> None:
    dataset = profile["dataset"]
    categories = profile["categories"]["items"][:20]
    identifiers = profile["identifiers"]["by_key"]
    clusters = profile["clusters"]
    kvp = profile["key_value_pairs"]
    quality = profile["quality"]

    lines = [
        "# selected_offers profile",
        "",
        f"- Source: `{dataset['source_file']}`",
        f"- Total records: {dataset['total_records']:,}",
        f"- Duration: {dataset['profiling_duration_seconds']}s",
        f"- Profiler: {dataset['profiler_version']}",
        "",
        "## Top categories",
        "",
    ]
    for row in categories:
        lines.append(
            f"- {row['category']}: {row['record_count']:,} "
            f"({row['percent_of_dataset']}%)"
        )

    lines.extend(["", "## Identifier summary", ""])
    for row in identifiers:
        lines.append(
            f"- {row['identifier_key']}: {row['record_count']:,} records "
            f"({row['percent_of_dataset']}%)"
        )

    lines.extend(
        [
            "",
            "## Cluster summary",
            "",
            f"- Unique clusters: {clusters['unique_cluster_count']:,}",
            f"- Singleton clusters: {clusters['singleton_cluster_count']:,}",
            f"- Multi-offer clusters: {clusters['multi_offer_cluster_count']:,}",
            f"- Max cluster size: {clusters['max_cluster_size']:,}",
            "",
            "## KVP summary",
            "",
            f"- Records with KVP: {kvp['records_with_key_value_pairs']:,}",
            f"- Unique attribute names: {kvp['unique_attribute_name_count']:,}",
            f"- Max attributes per record: {kvp['max_attribute_count']:,}",
            "",
            "## Quality signals",
            "",
            f"- Empty title: {quality['records_with_empty_title']:,}",
            f"- Without brand: {quality['records_without_brand']:,}",
            f"- Without description: {quality['records_without_description']:,}",
            f"- Without price: {quality['records_without_price']:,}",
            f"- Without category: {quality['records_without_category']:,}",
            "",
        ]
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@dataclass(frozen=True)
class ProfileResult:
    input_path: Path
    output_path: Path
    markdown_path: Path | None
    profile: dict[str, Any]
    peak_memory_bytes: int | None


def profile_selected_dataset(
    *,
    input_path: Path,
    output_path: Path,
    markdown_path: Path | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> ProfileResult:
    profiling_started_at = datetime.now(UTC)
    accumulator, peak_memory_bytes = profile_dataset(
        input_path=input_path,
        batch_size=batch_size,
    )
    profiling_completed_at = datetime.now(UTC)

    profile = build_profile_document(
        input_path=input_path,
        accumulator=accumulator,
        profiling_started_at=profiling_started_at,
        profiling_completed_at=profiling_completed_at,
        peak_memory_bytes=peak_memory_bytes,
    )
    write_profile_json(output_path, profile)
    if markdown_path is not None:
        write_profile_markdown(markdown_path, profile)

    return ProfileResult(
        input_path=input_path,
        output_path=output_path,
        markdown_path=markdown_path,
        profile=profile,
        peak_memory_bytes=peak_memory_bytes,
    )


def _format_duration(seconds: float) -> str:
    hours, remainder = divmod(int(seconds), 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def _print_summary(result: ProfileResult) -> None:
    profile = result.profile
    dataset = profile["dataset"]
    peak_memory_mb = (
        None
        if result.peak_memory_bytes is None
        else result.peak_memory_bytes / (1024 * 1024)
    )
    output_size = result.output_path.stat().st_size

    print(f"TOTAL RECORDS: {dataset['total_records']:,}")
    print(f"PROFILE BUILD TIME: {_format_duration(dataset['profiling_duration_seconds'])}")
    print(f"PROFILE FILE SIZE: {output_size:,} bytes")
    if peak_memory_mb is not None:
        print(f"PEAK MEMORY (traced): {peak_memory_mb:.1f} MiB")
    print(f"OUTPUT PATH: {result.output_path.resolve()}")
    if result.markdown_path is not None:
        print(f"MARKDOWN PATH: {result.markdown_path.resolve()}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Profile selected_offers.parquet with streaming single-pass statistics."
        ),
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to selected_offers.parquet.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to output profile JSON.",
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=None,
        help="Optional path to human-readable markdown summary.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Parquet read batch size (default: {DEFAULT_BATCH_SIZE:,}).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = profile_selected_dataset(
        input_path=args.input,
        output_path=args.output,
        markdown_path=args.markdown,
        batch_size=args.batch_size,
    )
    _print_summary(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
