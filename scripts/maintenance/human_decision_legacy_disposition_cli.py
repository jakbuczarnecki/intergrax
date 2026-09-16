#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""MP-4R6 — offline legacy human decision inventory and disposition CLI."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT, REPO_ROOT / "agents", REPO_ROOT / "applications"):
    path_value = str(path)
    if path_value not in sys.path:
        sys.path.insert(0, path_value)

from intergrax.contracts.human_decision_legacy_disposition import (  # noqa: E402
    HumanDecisionLegacyDispositionStrategy,
)
from intergrax.integrations.providers.relational_store.sqlite.human_decision_legacy import (  # noqa: E402
    assess_sqlite_human_decision_legacy_rows,
)
from intergrax.integrations.providers.relational_store.sqlite.paths import (  # noqa: E402
    resolve_human_decisions_db_path,
)
from intergrax.runtime.migration.human_decision_legacy_disposition import (  # noqa: E402
    export_sqlite_legacy_human_decision_archive_json,
    run_sqlite_human_decision_legacy_disposition,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db-path",
        type=Path,
        default=None,
        help="SQLite human decisions database (default: resolved platform path).",
    )
    parser.add_argument(
        "--strategy",
        choices=[item.value for item in HumanDecisionLegacyDispositionStrategy],
        default=HumanDecisionLegacyDispositionStrategy.HISTORY_ONLY_QUARANTINE.value,
        help="Offline disposition strategy (default: history-only quarantine report).",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply mutations (default is dry-run).",
    )
    parser.add_argument(
        "--allow-delete",
        action="store_true",
        help="Required for controlled delete strategy.",
    )
    parser.add_argument(
        "--export-archive-json",
        type=Path,
        default=None,
        help="Write non-authoritative legacy archive JSON to path.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    db_path = args.db_path or resolve_human_decisions_db_path()
    dry_run = not args.apply

    assessment = assess_sqlite_human_decision_legacy_rows(db_path)
    print(f"database: {db_path}")
    print(f"rows_total: {assessment.rows_total}")
    print(f"rows_with_approver_json: {assessment.rows_with_approver_json}")
    print(f"rows_missing_approver_json: {assessment.rows_missing_approver_json}")
    print(f"rows_malformed_approver_json: {assessment.rows_malformed_approver_json}")

    strategy = HumanDecisionLegacyDispositionStrategy(args.strategy)
    report = run_sqlite_human_decision_legacy_disposition(
        db_path,
        strategy=strategy,
        dry_run=dry_run,
        recovery_source=None,
        allow_delete=args.allow_delete,
    )
    print(f"dry_run: {report.dry_run}")
    print(f"rows_scanned: {report.rows_scanned}")
    print(f"rows_valid: {report.rows_valid}")
    print(f"rows_missing_provenance: {report.rows_missing_provenance}")
    print(f"rows_malformed_provenance: {report.rows_malformed_provenance}")
    print(f"rows_recoverable: {report.rows_recoverable}")
    print(f"rows_unrecoverable: {report.rows_unrecoverable}")
    print(f"rows_recovered: {report.rows_recovered}")
    print(f"rows_quarantined: {report.rows_quarantined}")
    print(f"rows_deleted: {report.rows_deleted}")

    if args.export_archive_json is not None:
        args.export_archive_json.parent.mkdir(parents=True, exist_ok=True)
        args.export_archive_json.write_text(
            export_sqlite_legacy_human_decision_archive_json(db_path),
            encoding="utf-8",
        )
        print(f"archive_export: {args.export_archive_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
