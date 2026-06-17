# © Artur Czarnecki. All rights reserved.
"""Initialize a dated architecture audit run folder and progress.json."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

from architecture_audit_common import (
    DOMAIN_ORDER,
    REPO_ROOT,
    RESULTS_ROOT,
    build_progress_template,
    resolve_bootstrap,
)

RUN_SUMMARY_STUB = """# Architecture audit run — {run_id}

**Mode:** {mode} · **Scope:** {scope}

## Status

Run initialized. Complete per-domain results under this directory.

## Rollup

| Domain | Verdict | P0 | P1 | Plan updated |
|--------|---------|----|----|--------------|
{rows}

## Notes

_(append operator or agent notes)_
"""


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--date",
        default=date.today().isoformat(),
        help="Run id folder name (default: today YYYY-MM-DD)",
    )
    parser.add_argument(
        "--mode",
        choices=("audit_only", "implement_plan", "layer_completion"),
        default="audit_only",
        help="Orchestration mode",
    )
    parser.add_argument(
        "--domain",
        default=None,
        help="Single domain basename (e.g. MEMORY). Omit for all 22 pairs.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing progress.json")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.domain and args.domain not in DOMAIN_ORDER:
        print(f"Unknown domain: {args.domain}", file=sys.stderr)
        print(f"Valid: {', '.join(DOMAIN_ORDER)}", file=sys.stderr)
        return 1

    run_dir = RESULTS_ROOT / args.date
    run_dir.mkdir(parents=True, exist_ok=True)
    progress_path = run_dir / "progress.json"

    if progress_path.exists() and not args.force:
        print(f"progress.json already exists: {progress_path}")
        print("Use --force to overwrite.")
        return 1

    scope = "single" if args.domain else "all"
    progress = build_progress_template(
        run_id=args.date,
        mode=args.mode,
        scope=scope,
        single_domain=args.domain,
    )
    progress_path.write_text(json.dumps(progress, indent=2) + "\n", encoding="utf-8")

    rows = "\n".join(f"| `{d}` | pending | — | — | — |" for d in progress["domain_order"])
    summary_path = run_dir / "RUN_SUMMARY.md"
    if not summary_path.exists() or args.force:
        summary_path.write_text(
            RUN_SUMMARY_STUB.format(
                run_id=args.date,
                mode=args.mode,
                scope=scope,
                rows=rows,
            ),
            encoding="utf-8",
        )

    bootstrap = resolve_bootstrap(args.mode, args.domain)
    print(f"Initialized run: {run_dir}")
    print(f"  progress: {progress_path.relative_to(REPO_ROOT)}")
    print(f"  paste bootstrap: {bootstrap}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
