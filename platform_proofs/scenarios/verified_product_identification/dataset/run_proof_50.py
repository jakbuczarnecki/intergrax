"""CLI entrypoint for proof-50 data pack build and validation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.paths import (
    DATASET_DIR,
    DEFAULT_PROOF_50_ROOT,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.status import (
    DataPackStatus,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.integration.proof_runner import (
    run_proof_50,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build and validate VPI proof-50 data pack")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_PROOF_50_ROOT,
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
        "--skip-rebuild",
        action="store_true",
        help="Reuse existing proof-50 artifacts and run validation/load/retrieval only",
    )
    args = parser.parse_args(argv)
    report = run_proof_50(
        output_root=args.output_root,
        dataset_path=args.dataset_path,
        dataset_manifest_path=args.dataset_manifest_path,
        rebuild_data_pack=not args.skip_rebuild,
    )
    print(f"proof-50 status: {report.status.value}")
    print(f"evidence: {args.output_root / 'evidence' / 'proof-report.json'}")
    return 0 if report.status is DataPackStatus.READY else 1


if __name__ == "__main__":
    raise SystemExit(main())
