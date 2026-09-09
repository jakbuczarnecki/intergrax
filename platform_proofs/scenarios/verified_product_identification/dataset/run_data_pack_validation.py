"""CLI entrypoint for full VPI Data Pack validation."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.validation.plan import (
    canonical_v1_validation_expectations,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.validation.report import (
    write_validation_report_json,
    write_validation_report_markdown,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.validation.service import (
    DataPackValidationPreconditionError,
    validate_full_data_pack,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.validation.contracts import (
    DataPackValidationVerdict,
)

_EXIT_PASS = 0
_EXIT_VALIDATION_FAIL = 1
_EXIT_PRECONDITION_FAIL = 2


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate finalized VPI Data Pack v1 artifact")
    parser.add_argument(
        "--artifact-root",
        type=Path,
        required=True,
        help="Root directory of finalized Data Pack artifact",
    )
    parser.add_argument(
        "--scratch-root",
        type=Path,
        default=None,
        help="Directory for bounded-memory duplicate-validation scratch files",
    )
    parser.add_argument(
        "--keep-scratch",
        action="store_true",
        help="Retain scratch files after validation completes",
    )
    parser.add_argument(
        "--report-root",
        type=Path,
        default=None,
        help="Directory for validation report outputs",
    )
    args = parser.parse_args(argv)
    _configure_logging()

    try:
        report = validate_full_data_pack(
            args.artifact_root,
            expectations=canonical_v1_validation_expectations(),
            scratch_root=args.scratch_root,
            keep_scratch=args.keep_scratch,
        )
    except DataPackValidationPreconditionError as exc:
        logging.error("precondition failed: %s", exc)
        return _EXIT_PRECONDITION_FAIL

    if args.report_root is not None:
        args.report_root.mkdir(parents=True, exist_ok=True)
        write_validation_report_json(
            args.report_root / "full-data-pack-validation-report.json",
            report,
        )
        write_validation_report_markdown(
            args.report_root / "FULL_DATA_PACK_VALIDATION_REPORT.md",
            report,
        )
        log_path = args.report_root / "validation.log"
        log_path.write_text(
            "\n".join(
                (
                    f"verdict={report.verdict.value}",
                    f"artifact_root={report.summary.artifact_root}",
                    f"expected_record_count={report.summary.expected_record_count}",
                    f"observed_relational_count={report.summary.observed_relational_count}",
                    f"observed_embedding_count={report.summary.observed_embedding_count}",
                    f"finalized_artifact_valid={'YES' if report.summary.finalized_artifact_valid else 'NO'}",
                )
            )
            + "\n",
            encoding="utf-8",
        )

    if report.verdict is DataPackValidationVerdict.PASS:
        logging.info("validation PASS artifact=%s", args.artifact_root)
        return _EXIT_PASS
    logging.error("validation FAIL artifact=%s", args.artifact_root)
    return _EXIT_VALIDATION_FAIL


if __name__ == "__main__":
    raise SystemExit(main())
