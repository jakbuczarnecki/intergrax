#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""Mechanical validator for ENT-AUDIT-X post-implementation closure master plan SSOT."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from master_plan_ssot_lib import (
    build_matrix_e_table,
    master_plan_path,
    parse_dag_table,
    parse_impl_qual_counts,
    parse_matrix_c,
    repo_root,
    topological_sort,
    validate_master_plan,
)


def _replace_section(text: str, heading: str, new_body: str) -> str:
    start_marker = heading
    idx = text.find(start_marker)
    if idx < 0:
        raise ValueError(f"heading not found: {heading}")
    after = text[idx + len(start_marker) :]
    next_idx = after.find("\n## ")
    if next_idx < 0:
        return text[:idx] + new_body
    return text[:idx] + new_body + after[next_idx:]


def run_validate(path: Path, write_json: Path | None) -> int:
    text = path.read_text(encoding="utf-8")
    errors = validate_master_plan(text, path.parents[4] if "docs" in str(path) else repo_root())
    matrix_c = parse_matrix_c(text)
    impl_counts, qual_counts = parse_impl_qual_counts(matrix_c)
    dag = parse_dag_table(text)
    try:
        topo = topological_sort(dag)
    except ValueError:
        topo = []

    report = {
        "path": str(path),
        "errors": errors,
        "pass": not errors,
        "matrix_c_impl": impl_counts,
        "matrix_c_qual": qual_counts,
        "topological_order_len": len(topo),
    }
    if write_json:
        write_json.parent.mkdir(parents=True, exist_ok=True)
        write_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if errors:
        for err in errors:
            print(err, file=sys.stderr)
        print(f"FAIL ({len(errors)} checks)", file=sys.stderr)
        return 1
    print("PASS")
    return 0


def regenerate_matrix_e(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    new_e = build_matrix_e_table()
    updated = _replace_section(text, "## Matrix E — 85 semantic oracles", new_e)
    path.write_text(updated, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        type=Path,
        default=None,
        help="Master plan markdown path",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Write validation JSON report",
    )
    parser.add_argument(
        "--regenerate-matrix-e",
        action="store_true",
        help="Replace Matrix E from manifest + Annex C",
    )
    args = parser.parse_args()
    path = args.path or master_plan_path()
    if args.regenerate_matrix_e:
        regenerate_matrix_e(path)
    json_out = args.json_out
    if json_out is None and not args.regenerate_matrix_e:
        json_out = repo_root() / ".tmp/session/ent-audit-x-plan-validator/plan-validator.json"
    return run_validate(path, json_out)


if __name__ == "__main__":
    raise SystemExit(main())
