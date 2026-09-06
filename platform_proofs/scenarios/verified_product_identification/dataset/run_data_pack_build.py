"""CLI entrypoint for resumable VPI Data Pack v1 builds."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from scripts.proof.intergrax_proof_environment import load_proof_environment

from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.application.resumable_builder import (
    DataPackBuildConfig,
    run_resumable_data_pack_build,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.paths import (
    DATASET_DIR,
    DEFAULT_CANONICAL_BUILD_ROOT,
    DEFAULT_PRODUCTION_SHARD_SIZE,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.status import (
    DataPackStatus,
)
from platform_proofs.scenarios.verified_product_identification.integrations.embedding.bootstrap import (
    ensure_embedding_provider_integrations_registered,
)

_SCENARIO_DIR = Path(__file__).resolve().parents[1]
_REPO_ROOT = _SCENARIO_DIR.parents[2]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build resumable VPI Data Pack v1")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_CANONICAL_BUILD_ROOT,
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=DATASET_DIR / "processed" / "selected_offers.parquet",
    )
    parser.add_argument(
        "--dataset-manifest-path",
        type=Path,
        default=DATASET_DIR / "processed" / "selected_offers_manifest.json",
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=DEFAULT_PRODUCTION_SHARD_SIZE,
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from state/build-state.json (required when build state exists)",
    )
    parser.add_argument(
        "--start-fresh",
        action="store_true",
        help="Clear scenario-owned build subtree and start a new full-plan build",
    )
    parser.add_argument(
        "--max-shards",
        type=int,
        default=None,
        help="Qualification only: stop after N shard ordinals",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help=(
            "Qualification/debug only: cap expected_record_count below manifest "
            "(changes build plan; omit for full selected-dataset build)"
        ),
    )
    parser.add_argument(
        "--stop-after-shard",
        type=int,
        default=None,
        help="Qualification only: graceful stop after shard ordinal N (full plan preserved)",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    load_proof_environment(proof_package_dir=_SCENARIO_DIR, repository_root=_REPO_ROOT)
    ensure_embedding_provider_integrations_registered()

    config = DataPackBuildConfig(
        output_root=args.output_root,
        dataset_path=args.dataset_path,
        dataset_manifest_path=args.dataset_manifest_path,
        shard_size=args.shard_size,
        resume=args.resume,
        start_fresh=args.start_fresh,
        max_shards=args.max_shards,
        max_records=args.max_records,
        stop_after_shard=args.stop_after_shard,
    )
    report = run_resumable_data_pack_build(config)
    progress = report.progress
    print(f"build status: {report.status.value}")
    print(
        f"progress: {progress.ready_shards}/{progress.total_shards} shards, "
        f"{progress.records_completed}/{progress.expected_records} records "
        f"({progress.percentage:.1f}%)"
    )
    if report.finalized:
        print(f"manifest: {args.output_root / 'manifest' / 'manifest.json'}")
    else:
        print("partial build — not distributable (no READY manifest)")
    return 0 if report.status is DataPackStatus.READY and report.finalized else 1


if __name__ == "__main__":
    raise SystemExit(main())
