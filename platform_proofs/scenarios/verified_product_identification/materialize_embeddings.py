"""VPI embedding artifact materialization CLI entry point."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from scripts.proof.intergrax_proof_environment import load_proof_environment

from platform_proofs.scenarios.verified_product_identification.composition.materialization_runtime import (
    build_vpi_embedding_materialization_runtime,
)
from platform_proofs.scenarios.verified_product_identification.embedding_materialization.contracts.config import (
    DEFAULT_MAX_RECORDS,
    load_vpi_embedding_materialization_config,
)
from platform_proofs.scenarios.verified_product_identification.embedding_materialization.manifest.model import (
    EmbeddingArtifactState,
)

_SCENARIO_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCENARIO_DIR.parents[2]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="VPI restartable embedding artifact materialization"
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=DEFAULT_MAX_RECORDS,
        help=f"Bounded materialization target (default: {DEFAULT_MAX_RECORDS})",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=None,
        help="Override artifact directory (default: fingerprinted path under VPI_EMBEDDING_ARTIFACT_PATH)",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate existing artifact without materializing new vectors",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    load_proof_environment(proof_package_dir=_SCENARIO_DIR, repository_root=_REPO_ROOT)

    config = load_vpi_embedding_materialization_config(
        max_records_override=args.max_records,
        artifact_dir_override=args.artifact_dir,
    )
    orchestrator = build_vpi_embedding_materialization_runtime(
        config,
        artifact_dir=args.artifact_dir,
    )
    try:
        report = orchestrator.run(validate_only=args.validate_only)
    finally:
        orchestrator.dependencies.artifact_writer.close()
        orchestrator.dependencies.embedding.close()

    print(f"state={report.final_state.value}")
    if report.manifest is not None:
        print(
            "rows="
            f"{report.manifest.checkpoint_rows_materialized}"
            f" shards={report.manifest.shard_count}"
        )
    print(
        "timing="
        f"total={report.elapsed_total_seconds:.3f}s "
        f"embed={report.elapsed_embedding_seconds:.3f}s "
        f"derive={report.elapsed_derive_seconds:.3f}s "
        f"write={report.elapsed_artifact_write_seconds:.3f}s "
        f"embed_calls={report.embedding_calls}"
    )
    if report.effective_records_per_second > 0:
        print(f"records_per_sec={report.effective_records_per_second:.2f}")
    if report.failure_detail:
        print(f"failure={report.failure_detail}", file=sys.stderr)

    return 0 if report.final_state is EmbeddingArtifactState.READY else 1


if __name__ == "__main__":
    raise SystemExit(main())
