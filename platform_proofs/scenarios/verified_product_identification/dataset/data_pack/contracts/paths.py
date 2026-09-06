"""Canonical filesystem layout for VPI data packs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

DATASET_DIR = Path(__file__).resolve().parents[2]
DEFAULT_GENERATED_ROOT = DATASET_DIR / "generated" / "data_pack"
DEFAULT_PROOF_50_ROOT = DEFAULT_GENERATED_ROOT / "proof-50"
DEFAULT_CANONICAL_BUILD_ROOT = DEFAULT_GENERATED_ROOT / "canonical-v1"
DEFAULT_PRODUCTION_SHARD_SIZE = 25_000
PROOF_50_POSTGRESQL_SCHEMA = "vpi_proof_5c4d1"
PROOF_50_QDRANT_COLLECTION = "vpi_offers_proof_5c4d1"


@dataclass(frozen=True, slots=True)
class DataPackPaths:
    root: Path
    manifest_dir: Path
    manifest_file: Path
    relational_dir: Path
    embeddings_dir: Path
    indexes_dir: Path
    checksums_dir: Path
    evidence_dir: Path
    state_dir: Path
    build_state_file: Path
    shards_index_file: Path
    checksums_file: Path
    proof_report_file: Path


def resolve_data_pack_paths(root: Path) -> DataPackPaths:
    return DataPackPaths(
        root=root,
        manifest_dir=root / "manifest",
        manifest_file=root / "manifest" / "manifest.json",
        relational_dir=root / "relational",
        embeddings_dir=root / "embeddings",
        indexes_dir=root / "indexes",
        checksums_dir=root / "checksums",
        evidence_dir=root / "evidence",
        state_dir=root / "state",
        build_state_file=root / "state" / "build-state.json",
        shards_index_file=root / "indexes" / "shards.json",
        checksums_file=root / "checksums" / "SHA256SUMS",
        proof_report_file=root / "evidence" / "proof-report.json",
    )


def shard_file_name(shard_ordinal: int) -> str:
    return f"part-{shard_ordinal:06d}.parquet"


def temp_shard_path(directory: Path, shard_ordinal: int) -> Path:
    return directory / f"{shard_file_name(shard_ordinal)}.tmp"


def final_shard_path(directory: Path, shard_ordinal: int) -> Path:
    return directory / shard_file_name(shard_ordinal)
