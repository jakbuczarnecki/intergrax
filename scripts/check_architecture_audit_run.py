# © Artur Czarnecki. All rights reserved.
"""Validate architecture audit run folder structure and progress.json."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from architecture_audit_common import DOMAIN_ORDER, REPO_ROOT, RESULTS_ROOT

REQUIRED_PROGRESS_KEYS = {
    "orchestrator",
    "mode",
    "scope",
    "canonical",
    "bootstrap",
    "run_id",
    "results_dir",
    "started_at",
    "current_domain",
    "domain_order",
    "domains",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run_id",
        nargs="?",
        default=None,
        help="Run folder name (YYYY-MM-DD). Default: latest under results/",
    )
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Fail unless every domain status is completed, skipped, or blocked",
    )
    return parser.parse_args()


def _resolve_run_dir(run_id: str | None) -> Path | None:
    if run_id:
        path = RESULTS_ROOT / run_id
        return path if path.is_dir() else None
    if not RESULTS_ROOT.is_dir():
        return None
    candidates = sorted(
        (p for p in RESULTS_ROOT.iterdir() if p.is_dir() and (p / "progress.json").is_file()),
        key=lambda p: p.name,
        reverse=True,
    )
    return candidates[0] if candidates else None


def main() -> int:
    args = _parse_args()
    run_dir = _resolve_run_dir(args.run_id)
    if run_dir is None:
        print("check_architecture_audit_run: no run directory found", file=sys.stderr)
        return 1

    progress_path = run_dir / "progress.json"
    try:
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"Invalid JSON: {progress_path}: {exc}", file=sys.stderr)
        return 1

    missing = REQUIRED_PROGRESS_KEYS - set(progress)
    if missing:
        print(f"progress.json missing keys: {sorted(missing)}", file=sys.stderr)
        return 1

    errors: list[str] = []
    domains = progress.get("domains", {})
    if not isinstance(domains, dict):
        print("domains must be an object", file=sys.stderr)
        return 1

    for domain in progress.get("domain_order", []):
        if domain not in DOMAIN_ORDER:
            errors.append(f"unknown domain in domain_order: {domain}")
        entry = domains.get(domain)
        if not isinstance(entry, dict):
            errors.append(f"missing domains.{domain}")
            continue
        result_md = entry.get("result_md")
        status = entry.get("status")
        if status == "completed":
            if not result_md:
                errors.append(f"{domain}: completed but no result_md")
            else:
                result_path = Path(str(result_md))
                if not result_path.is_absolute():
                    result_path = REPO_ROOT / result_path
                if not result_path.is_file():
                    errors.append(f"{domain}: missing result file {result_md}")
        if args.require_complete and status not in ("completed", "skipped", "blocked"):
            errors.append(f"{domain}: status is {status!r}, expected completed/skipped/blocked")

    summary = run_dir / "RUN_SUMMARY.md"
    if not summary.is_file():
        errors.append("missing RUN_SUMMARY.md")

    if errors:
        for line in errors:
            print(f"  - {line}", file=sys.stderr)
        print(f"check_architecture_audit_run: FAIL ({run_dir.name})", file=sys.stderr)
        return 1

    print(f"check_architecture_audit_run: OK ({run_dir.name}, {len(domains)} domains)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
