"""VPI storage bootstrap CLI entry point."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from scripts.proof.intergrax_proof_environment import load_proof_environment

from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.contracts.config import (
    BootstrapRunMode,
    load_vpi_bootstrap_config,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.manifest.model import (
    BootstrapState,
)
from platform_proofs.scenarios.verified_product_identification.composition.bootstrap_runtime import (
    build_vpi_bootstrap_runtime,
)

_SCENARIO_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCENARIO_DIR.parents[2]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="VPI provider-neutral storage bootstrap")
    parser.add_argument(
        "--mode",
        choices=[mode.value for mode in BootstrapRunMode],
        default=BootstrapRunMode.VERIFY.value,
        help="verify uses bounded ingest; full uses configured max_records (unlimited when unset)",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Override bounded ingest record count (verify mode default: 1000)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    load_proof_environment(proof_package_dir=_SCENARIO_DIR, repository_root=_REPO_ROOT)

    mode = BootstrapRunMode(args.mode)
    config = load_vpi_bootstrap_config(mode=mode, max_records_override=args.max_records)
    orchestrator = build_vpi_bootstrap_runtime(config)
    try:
        report = orchestrator.run()
    finally:
        orchestrator.dependencies.catalog.close()
        orchestrator.dependencies.search.close()

    print(f"state={report.final_state.value}")
    if report.manifest is not None:
        print(
            "rows="
            f"{report.manifest.checkpoint_rows_processed}"
            f" offers={report.manifest.catalog_source_offer_count}"
            f" points={report.manifest.search_point_count}"
        )
    if report.failure_detail:
        print(f"failure={report.failure_detail}", file=sys.stderr)

    return 0 if report.final_state is BootstrapState.READY else 1


if __name__ == "__main__":
    raise SystemExit(main())
