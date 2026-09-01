"""Sample a fixed number of records from selected_offers.parquet."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TypeAlias

import pyarrow as pa
import pyarrow.parquet as pq

SAMPLER_VERSION = "verified_product_identification_wdc_sampler/1.0.0"
DATASET_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_PATH = DATASET_DIR / "processed" / "selected_offers.parquet"
DEFAULT_SAMPLE_SIZE = 1_000
DEFAULT_BATCH_SIZE = 10_000

JsonPrimitive: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonPrimitive | list["JsonValue"] | dict[str, "JsonValue"]

PARQUET_SCHEMA = pa.schema([("record_json", pa.string())])


@dataclass(frozen=True)
class SampleResult:
    input_path: Path
    output_path: Path
    manifest_path: Path
    source_record_count: int
    sample_size: int
    sampled_record_count: int
    output_size_bytes: int
    output_sha256: str
    random_seed: int
    sample_started_at: datetime
    sample_completed_at: datetime


def default_output_path(input_path: Path, sample_size: int) -> Path:
    return input_path.with_name(f"{input_path.stem}_sample_{sample_size}.parquet")


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


def reservoir_sample(
    records: Iterator[str],
    *,
    sample_size: int,
    rng: random.Random,
) -> tuple[list[str], int]:
    reservoir: list[str] = []
    source_record_count = 0

    for record in records:
        if source_record_count < sample_size:
            reservoir.append(record)
        else:
            replacement_index = rng.randint(0, source_record_count)
            if replacement_index < sample_size:
                reservoir[replacement_index] = record
        source_record_count += 1

    return reservoir, source_record_count


def write_sample_parquet(output_path: Path, records: list[str]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pq.ParquetWriter(
        where=str(output_path),
        schema=PARQUET_SCHEMA,
        compression="zstd",
    ) as writer:
        table = pa.Table.from_arrays(
            [pa.array(records, type=pa.string())],
            schema=PARQUET_SCHEMA,
        )
        writer.write_table(table)


def build_manifest(
    *,
    result: SampleResult,
) -> dict[str, JsonValue]:
    return {
        "sampler_version": SAMPLER_VERSION,
        "source_path": str(result.input_path.resolve()),
        "source_record_count": result.source_record_count,
        "requested_sample_size": result.sample_size,
        "sampled_record_count": result.sampled_record_count,
        "random_seed": result.random_seed,
        "sampling_method": (
            "uniform reservoir sampling over record_json rows "
            "(single pass, bounded memory)"
        ),
        "output_format": "parquet",
        "compression": "zstd",
        "output_path": str(result.output_path.resolve()),
        "output_size_bytes": result.output_size_bytes,
        "output_sha256": result.output_sha256,
        "parquet_representation": {
            "columns": ["record_json"],
            "nested_encoding": (
                "Each sampled record is copied losslessly from the source Parquet "
                "record_json column."
            ),
        },
        "sample_started_at": result.sample_started_at.isoformat(),
        "sample_completed_at": result.sample_completed_at.isoformat(),
    }


def write_manifest(path: Path, manifest: dict[str, JsonValue]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def sample_dataset(
    *,
    input_path: Path,
    output_path: Path | None = None,
    manifest_path: Path | None = None,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    random_seed: int = 42,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> SampleResult:
    if sample_size <= 0:
        msg = "sample_size must be positive"
        raise ValueError(msg)
    if batch_size <= 0:
        msg = "batch_size must be positive"
        raise ValueError(msg)
    if not input_path.is_file():
        msg = f"input file does not exist: {input_path}"
        raise FileNotFoundError(msg)

    resolved_output_path = (
        output_path
        if output_path is not None
        else default_output_path(input_path, sample_size)
    )
    resolved_manifest_path = (
        manifest_path
        if manifest_path is not None
        else default_manifest_path(resolved_output_path)
    )

    rng = random.Random(random_seed)
    sample_started_at = datetime.now(UTC)

    sampled_records, source_record_count = reservoir_sample(
        iter_record_json(input_path, batch_size=batch_size),
        sample_size=sample_size,
        rng=rng,
    )

    write_sample_parquet(resolved_output_path, sampled_records)

    sample_completed_at = datetime.now(UTC)
    output_size_bytes = resolved_output_path.stat().st_size
    output_sha256 = sha256_file(resolved_output_path)

    result = SampleResult(
        input_path=input_path,
        output_path=resolved_output_path,
        manifest_path=resolved_manifest_path,
        source_record_count=source_record_count,
        sample_size=sample_size,
        sampled_record_count=len(sampled_records),
        output_size_bytes=output_size_bytes,
        output_sha256=output_sha256,
        random_seed=random_seed,
        sample_started_at=sample_started_at,
        sample_completed_at=sample_completed_at,
    )
    write_manifest(resolved_manifest_path, build_manifest(result=result))
    return result


def _format_duration(seconds: float) -> str:
    hours, remainder = divmod(int(seconds), 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def _print_summary(result: SampleResult) -> None:
    sample_seconds = (
        result.sample_completed_at - result.sample_started_at
    ).total_seconds()

    print(f"SOURCE RECORDS: {result.source_record_count:,}")
    print(f"REQUESTED SAMPLE SIZE: {result.sample_size:,}")
    print(f"SAMPLED RECORDS: {result.sampled_record_count:,}")
    print(f"RANDOM SEED: {result.random_seed}")
    print("OUTPUT FORMAT: parquet (zstd, record_json column)")
    print(f"OUTPUT FILE SIZE: {result.output_size_bytes:,} bytes")
    print(f"OUTPUT SHA256: {result.output_sha256}")
    print(f"SAMPLE TIME: {_format_duration(sample_seconds)}")
    print(f"OUTPUT PATH: {result.output_path.resolve()}")
    print(f"MANIFEST PATH: {result.manifest_path.resolve()}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sample a fixed number of records from selected_offers.parquet "
            "using streaming reservoir sampling."
        ),
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help=(
            "Path to source Parquet file "
            f"(default: {DEFAULT_INPUT_PATH.relative_to(DATASET_DIR)})."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Path to output Parquet file "
            "(default: <input_stem>_sample_<size>.parquet next to input)."
        ),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help=(
            "Manifest JSON path "
            "(default: <output_stem>_manifest.json next to output)."
        ),
    )
    parser.add_argument(
        "--size",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help=f"Number of records to sample (default: {DEFAULT_SAMPLE_SIZE:,}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling (default: 42).",
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
    started = time.perf_counter()
    result = sample_dataset(
        input_path=args.input,
        output_path=args.output,
        manifest_path=args.manifest,
        sample_size=args.size,
        random_seed=args.seed,
        batch_size=args.batch_size,
    )
    _ = started
    _print_summary(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
