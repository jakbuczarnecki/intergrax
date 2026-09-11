"""Diagnostic execution path for VPI embedding performance qualification (VPI-IMPLEMENTATION-5C4E2)."""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[5]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.application.embedding_diagnostics import (
    DIAGNOSTIC_MAX_RECORD_LIMIT,
    run_embedding_diagnostics,
    write_embedding_diagnostic_report,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.paths import (
    DATASET_DIR,
)


def _default_dataset_path() -> Path:
    return DATASET_DIR / "processed" / "selected_offers.parquet"


def _default_output_dir() -> Path:
    return _REPO_ROOT / ".tmp" / "session" / "vpi-5c4e2" / "embedding-diagnostics"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run VPI embedding diagnostics on a bounded dataset sample.",
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=_default_dataset_path(),
        help="Path to selected_offers.parquet",
    )
    parser.add_argument(
        "--record-limit",
        type=int,
        default=100,
        help=f"Maximum records to analyze (<= {DIAGNOSTIC_MAX_RECORD_LIMIT})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override production baseline batch size for experiment A",
    )
    parser.add_argument(
        "--experiment",
        default="full",
        help=(
            "Experiment scope: full, production_baseline, batch_32, batch_64, "
            "representation_only, A, B, C, D"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_default_output_dir(),
        help="Directory for machine-readable diagnostic evidence",
    )
    parser.add_argument(
        "--qualification-id",
        default=f"vpi-embedding-diagnostics-{uuid.uuid4().hex[:12]}",
        help="Stable qualification identifier for the diagnostic run",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if not args.dataset_path.is_file():
        raise SystemExit(f"dataset not found: {args.dataset_path}")
    report = run_embedding_diagnostics(
        dataset_path=args.dataset_path,
        record_limit=args.record_limit,
        qualification_id=args.qualification_id,
        production_batch_size=args.batch_size,
        experiment=args.experiment,
    )
    json_path, summary_path = write_embedding_diagnostic_report(args.output_dir, report)
    evidence = {
        "status": "PASS",
        "qualification_id": report.qualification_id,
        "record_limit": report.record_limit,
        "classification_case": report.classification.case.value,
        "recommended_next_task": report.classification.recommended_next_task,
        "baseline_records_per_second": round(report.baseline.records_per_second, 3),
        "token_p95": round(report.token_distribution.statistics.p95, 3),
        "report_json": str(json_path),
        "report_summary": str(summary_path),
    }
    print(json.dumps(evidence, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
