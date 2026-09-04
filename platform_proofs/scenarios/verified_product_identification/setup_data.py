"""Install the VPI portable data package."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from scripts.proof.intergrax_proof_environment import load_proof_environment

from platform_proofs.scenarios.verified_product_identification.data_package.config import (
    load_vpi_data_package_config,
)
from platform_proofs.scenarios.verified_product_identification.data_package.errors import (
    VpiDataPackageError,
)
from platform_proofs.scenarios.verified_product_identification.data_package.install import (
    install_vpi_data_package,
)

_SCENARIO_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCENARIO_DIR.parents[2]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download and verify the Verified Product Identification data package",
    )
    parser.add_argument(
        "--local-mirror",
        type=Path,
        default=None,
        help="Trusted local directory mirror (file:// transport) for offline/dev installs",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    load_proof_environment(proof_package_dir=_SCENARIO_DIR, repository_root=_REPO_ROOT)
    config = load_vpi_data_package_config()
    try:
        result = install_vpi_data_package(
            config,
            local_mirror_root=args.local_mirror,
        )
    except VpiDataPackageError as exc:
        print(f"error={exc}", file=sys.stderr)
        return 1

    report = result.install_report
    validation = result.validation_report
    print(f"package_id={report.package_id}")
    print(f"package_version={report.package_version}")
    print(f"install_location={report.install_location}")
    print(f"files_total={report.files_total}")
    print(f"files_downloaded={report.files_downloaded}")
    print(f"files_reused_from_cache={report.files_reused_from_cache}")
    print(f"files_installed_from_existing={report.files_installed_from_existing}")
    print(f"bytes_downloaded={report.bytes_downloaded}")
    print(f"bytes_reused={report.bytes_reused}")
    print(f"elapsed_seconds={report.elapsed_seconds:.3f}")
    print(f"dataset_checksum={validation.dataset_checksum}")
    print(f"dataset_record_count={validation.dataset_record_count}")
    print(f"embedding_model={validation.embedding_model}")
    print(f"redistribution_status={validation.redistribution_status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
