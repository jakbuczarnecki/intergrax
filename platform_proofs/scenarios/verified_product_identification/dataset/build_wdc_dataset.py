"""Streaming WDC dataset builder for verified_product_identification scenario."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import tracemalloc
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TypeAlias

import pyarrow as pa
import pyarrow.parquet as pq

BUILDER_VERSION = "verified_product_identification_wdc_builder/1.0.0"
SOURCE_DATASET_NAME = "offers_corpus_all_v2_non_norm"
SELECTION_RULE = (
    "keyValuePairs != null OR specTableContent != null"
)
DEFAULT_BATCH_SIZE = 5_000
PROGRESS_INTERVAL_SOURCE_RECORDS = 500_000

JsonPrimitive: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonPrimitive | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject: TypeAlias = dict[str, JsonValue]

PARQUET_SCHEMA = pa.schema([("record_json", pa.string())])


@dataclass(frozen=True)
class BuildStats:
    source_record_count: int
    selected_record_count: int
    malformed_record_count: int
    records_with_key_value_pairs: int
    records_with_spec_table_content: int
    records_with_both: int

    @property
    def rejected_record_count(self) -> int:
        return (
            self.source_record_count
            - self.selected_record_count
            - self.malformed_record_count
        )


@dataclass(frozen=True)
class BuildResult:
    stats: BuildStats
    output_path: Path
    manifest_path: Path
    output_size_bytes: int
    output_sha256: str
    build_started_at: datetime
    build_completed_at: datetime
    peak_memory_bytes: int | None


def parse_json_object(line: str) -> JsonObject:
    parsed = json.loads(line)
    if not isinstance(parsed, dict):
        msg = "top-level JSON value must be an object"
        raise ValueError(msg)
    return _normalize_json_object(parsed)


def _normalize_json_object(value: dict[object, object]) -> JsonObject:
    normalized: JsonObject = {}
    for raw_key, raw_value in value.items():
        if not isinstance(raw_key, str):
            msg = "JSON object keys must be strings"
            raise ValueError(msg)
        normalized[raw_key] = _normalize_json_value(raw_value)
    return normalized


def _normalize_json_value(value: object) -> JsonValue:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [_normalize_json_value(item) for item in value]
    if isinstance(value, dict):
        return _normalize_json_object(value)
    msg = f"unsupported JSON value type: {type(value).__name__}"
    raise ValueError(msg)


def record_is_selected(record: JsonObject) -> bool:
    return (
        record.get("keyValuePairs") is not None
        or record.get("specTableContent") is not None
    )


def selection_flags(record: JsonObject) -> tuple[bool, bool]:
    has_key_value_pairs = record.get("keyValuePairs") is not None
    has_spec_table_content = record.get("specTableContent") is not None
    return has_key_value_pairs, has_spec_table_content


def serialize_record(record: JsonObject) -> str:
    return json.dumps(record, ensure_ascii=False, separators=(",", ":"))


def default_manifest_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}_manifest.json")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(
    *,
    source_path: Path,
    output_path: Path,
    stats: BuildStats,
    output_size_bytes: int,
    output_sha256: str,
    build_started_at: datetime,
    build_completed_at: datetime,
) -> dict[str, JsonValue]:
    return {
        "builder_version": BUILDER_VERSION,
        "source_dataset_name": SOURCE_DATASET_NAME,
        "source_path": str(source_path.resolve()),
        "selection_rule": SELECTION_RULE,
        "source_record_count": stats.source_record_count,
        "selected_record_count": stats.selected_record_count,
        "rejected_record_count": stats.rejected_record_count,
        "malformed_record_count": stats.malformed_record_count,
        "records_with_key_value_pairs": stats.records_with_key_value_pairs,
        "records_with_spec_table_content": stats.records_with_spec_table_content,
        "records_with_both": stats.records_with_both,
        "unique_cluster_count": None,
        "unique_cluster_count_skipped_reason": (
            "Counting unique cluster_id values for millions of selected records "
            "requires unbounded memory; skipped for this offline builder."
        ),
        "output_format": "parquet",
        "compression": "zstd",
        "output_path": str(output_path.resolve()),
        "output_size_bytes": output_size_bytes,
        "output_sha256": output_sha256,
        "parquet_representation": {
            "columns": ["record_json"],
            "nested_encoding": (
                "Each selected source record is stored losslessly as one UTF-8 JSON "
                "string in record_json. This avoids unstable Parquet schema inference "
                "on heterogeneous top-level and nested fields."
            ),
        },
        "build_started_at": build_started_at.isoformat(),
        "build_completed_at": build_completed_at.isoformat(),
    }


def write_manifest(path: Path, manifest: dict[str, JsonValue]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_batch(writer: pq.ParquetWriter, batch: list[str]) -> None:
    table = pa.Table.from_arrays(
        [pa.array(batch, type=pa.string())],
        schema=PARQUET_SCHEMA,
    )
    writer.write_table(table)


def build_dataset(
    *,
    input_path: Path,
    output_path: Path,
    manifest_path: Path | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> BuildResult:
    if batch_size <= 0:
        msg = "batch_size must be positive"
        raise ValueError(msg)

    resolved_manifest_path = (
        manifest_path if manifest_path is not None else default_manifest_path(output_path)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    source_record_count = 0
    selected_record_count = 0
    malformed_record_count = 0
    records_with_key_value_pairs = 0
    records_with_spec_table_content = 0
    records_with_both = 0

    batch: list[str] = []
    tracemalloc.start()
    build_started_at = datetime.now(UTC)

    with input_path.open("r", encoding="utf-8") as source, pq.ParquetWriter(
        where=str(output_path),
        schema=PARQUET_SCHEMA,
        compression="zstd",
    ) as writer:
        for raw_line in source:
            source_record_count += 1
            line = raw_line.strip()
            if not line:
                malformed_record_count += 1
                continue

            try:
                record = parse_json_object(line)
            except (json.JSONDecodeError, ValueError, TypeError):
                malformed_record_count += 1
                continue

            if not record_is_selected(record):
                continue

            has_key_value_pairs, has_spec_table_content = selection_flags(record)
            if has_key_value_pairs:
                records_with_key_value_pairs += 1
            if has_spec_table_content:
                records_with_spec_table_content += 1
            if has_key_value_pairs and has_spec_table_content:
                records_with_both += 1

            batch.append(serialize_record(record))
            selected_record_count += 1

            if len(batch) >= batch_size:
                _write_batch(writer, batch)
                batch.clear()

            if source_record_count % PROGRESS_INTERVAL_SOURCE_RECORDS == 0:
                print(
                    (
                        f"[progress] source={source_record_count:,} "
                        f"selected={selected_record_count:,} "
                        f"malformed={malformed_record_count:,}"
                    ),
                    file=sys.stderr,
                )

        if batch:
            _write_batch(writer, batch)

    _, peak_memory_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    build_completed_at = datetime.now(UTC)
    output_size_bytes = output_path.stat().st_size
    output_sha256 = sha256_file(output_path)

    stats = BuildStats(
        source_record_count=source_record_count,
        selected_record_count=selected_record_count,
        malformed_record_count=malformed_record_count,
        records_with_key_value_pairs=records_with_key_value_pairs,
        records_with_spec_table_content=records_with_spec_table_content,
        records_with_both=records_with_both,
    )
    manifest = build_manifest(
        source_path=input_path,
        output_path=output_path,
        stats=stats,
        output_size_bytes=output_size_bytes,
        output_sha256=output_sha256,
        build_started_at=build_started_at,
        build_completed_at=build_completed_at,
    )
    write_manifest(resolved_manifest_path, manifest)

    return BuildResult(
        stats=stats,
        output_path=output_path,
        manifest_path=resolved_manifest_path,
        output_size_bytes=output_size_bytes,
        output_sha256=output_sha256,
        build_started_at=build_started_at,
        build_completed_at=build_completed_at,
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


def _print_summary(result: BuildResult) -> None:
    stats = result.stats
    build_seconds = (
        result.build_completed_at - result.build_started_at
    ).total_seconds()
    peak_memory_mb = (
        None
        if result.peak_memory_bytes is None
        else result.peak_memory_bytes / (1024 * 1024)
    )

    print(f"SOURCE RECORDS: {stats.source_record_count:,}")
    print(f"SELECTED RECORDS: {stats.selected_record_count:,}")
    print(f"REJECTED RECORDS: {stats.rejected_record_count:,}")
    print(f"MALFORMED: {stats.malformed_record_count:,}")
    print(f"WITH keyValuePairs: {stats.records_with_key_value_pairs:,}")
    print(f"WITH specTableContent: {stats.records_with_spec_table_content:,}")
    print(f"WITH BOTH: {stats.records_with_both:,}")
    print("OUTPUT FORMAT: parquet (zstd, record_json column)")
    print(f"OUTPUT FILE SIZE: {result.output_size_bytes:,} bytes")
    print(f"OUTPUT SHA256: {result.output_sha256}")
    print(f"BUILD TIME: {_format_duration(build_seconds)}")
    if peak_memory_mb is not None:
        print(f"PEAK MEMORY (traced): {peak_memory_mb:.1f} MiB")
    print(f"OUTPUT PATH: {result.output_path.resolve()}")
    print(f"MANIFEST PATH: {result.manifest_path.resolve()}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build verified_product_identification dataset from WDC NDJSON "
            "with streaming selection and ZSTD Parquet output."
        ),
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Path to source WDC NDJSON file.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Path to output Parquet file.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional manifest JSON path (defaults to <output_stem>_manifest.json).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    started = time.perf_counter()
    result = build_dataset(
        input_path=args.input,
        output_path=args.output,
        manifest_path=args.manifest,
    )
    _ = started
    _print_summary(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
