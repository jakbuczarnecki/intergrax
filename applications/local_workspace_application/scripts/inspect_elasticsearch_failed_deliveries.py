#!/usr/bin/env python3
"""Inspect persisted Local Workspace Elasticsearch failed-delivery JSONL records.

Lightweight readback tooling for the LKW Elasticsearch failed-delivery proof path.
Validates safe ``ElasticsearchFailedDeliveryRecord`` fields only; does not parse or
print raw documents, prompts, chunks, tool args, secrets, tokens, or payload paths.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import fields
from pathlib import Path
from typing import Any, Sequence

from intergrax.integrations.providers.observability_backend.elasticsearch.transport import (
    ElasticsearchFailedDeliveryRecord,
)

Record = dict[str, Any]
ValidationIssue = tuple[int, str]


SAFE_FIELD_NAMES = frozenset(field.name for field in fields(ElasticsearchFailedDeliveryRecord))

DISPLAY_FIELDS = (
    "provider_id",
    "operation",
    "index",
    "status_code",
    "reason",
    "retriable",
    "attempts",
    "exhausted",
)


def _validate_field_type(field_name: str, value: Any) -> str | None:
    if field_name == "status_code":
        if value is None or isinstance(value, int):
            return None
        return "status_code must be an integer or null"

    if field_name in {"provider_id", "operation", "index", "reason"}:
        if isinstance(value, str):
            return None
        return f"{field_name} must be a string"

    if field_name in {"retriable", "exhausted"}:
        if isinstance(value, bool):
            return None
        return f"{field_name} must be a boolean"

    if field_name == "attempts":
        if isinstance(value, int) and not isinstance(value, bool):
            return None
        return "attempts must be an integer"

    return f"unknown field: {field_name}"


def validate_record_object(obj: Any, *, line_no: int) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []

    if not isinstance(obj, dict):
        return [(line_no, "record must be a JSON object")]

    keys = set(obj.keys())
    extra = sorted(keys - SAFE_FIELD_NAMES)
    missing = sorted(SAFE_FIELD_NAMES - keys)
    if extra:
        issues.append((line_no, f"unexpected keys: {', '.join(extra)}"))
    if missing:
        issues.append((line_no, f"missing keys: {', '.join(missing)}"))
    if extra or missing:
        return issues

    for field_name in DISPLAY_FIELDS:
        type_issue = _validate_field_type(field_name, obj.get(field_name))
        if type_issue is not None:
            issues.append((line_no, type_issue))

    return issues


def load_records(path: Path) -> tuple[list[Record], list[ValidationIssue]]:
    records: list[Record] = []
    issues: list[ValidationIssue] = []

    with path.open(encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue

            try:
                parsed = json.loads(line)
            except json.JSONDecodeError as exc:
                issues.append((line_no, f"invalid JSON: {exc.msg}"))
                continue

            line_issues = validate_record_object(parsed, line_no=line_no)
            issues.extend(line_issues)
            if not line_issues:
                record = {field: parsed[field] for field in DISPLAY_FIELDS}
                record["line_no"] = line_no
                records.append(record)

    return records, issues


def summarize_records(records: Sequence[Record]) -> dict[str, Any]:
    reason_counts = Counter(str(record.get("reason") or "") for record in records)
    retriable_count = sum(1 for record in records if record.get("retriable") is True)
    exhausted_count = sum(1 for record in records if record.get("exhausted") is True)
    status_counts = Counter(
        "null" if record.get("status_code") is None else str(record.get("status_code"))
        for record in records
    )

    return {
        "record_count": len(records),
        "retriable_count": retriable_count,
        "exhausted_count": exhausted_count,
        "reason_counts": dict(reason_counts),
        "status_code_counts": dict(status_counts),
    }


def _truncate(value: Any, width: int) -> str:
    text = "" if value is None else str(value)
    if len(text) <= width:
        return text
    if width <= 1:
        return text[:width]
    return text[: width - 1] + "…"


def print_summary(path: Path, records: Sequence[Record], issues: Sequence[ValidationIssue]) -> None:
    summary = summarize_records(records)

    print(f"File: {path}")
    print(f"Records: {summary['record_count']}")
    print(f"Retriable: {summary['retriable_count']}")
    print(f"Exhausted: {summary['exhausted_count']}")

    if summary["reason_counts"]:
        print("Reason counts:")
        for reason, count in sorted(summary["reason_counts"].items()):
            print(f"  {reason}: {count}")

    if summary["status_code_counts"]:
        print("Status code counts:")
        for status_code, count in sorted(summary["status_code_counts"].items()):
            print(f"  {status_code}: {count}")

    if issues:
        print(f"Validation issues: {len(issues)}")
        for line_no, message in issues:
            print(f"  line {line_no}: {message}")
    else:
        print("Validation: all records contain exactly the safe failed-delivery fields")

    if records:
        columns = [
            ("line_no", "line", 5),
            ("provider_id", "provider", 12),
            ("operation", "operation", 24),
            ("index", "index", 24),
            ("status_code", "status", 8),
            ("reason", "reason", 20),
            ("retriable", "retry", 6),
            ("attempts", "attempts", 8),
            ("exhausted", "exhausted", 9),
        ]
        print()
        print(" ".join(f"{title:<{width}}" for _field, title, width in columns))
        print(" ".join("-" * width for _field, _title, width in columns))
        for record in records:
            print(
                " ".join(
                    f"{_truncate(record.get(field, ''), width):<{width}}"
                    for field, _title, width in columns
                )
            )


def to_json_payload(
    path: Path,
    records: Sequence[Record],
    issues: Sequence[ValidationIssue],
) -> dict[str, Any]:
    payload = summarize_records(records)
    payload["file"] = str(path)
    payload["records"] = [{field: record.get(field) for field in DISPLAY_FIELDS} for record in records]
    payload["validation_issues"] = [{"line_no": line_no, "message": message} for line_no, message in issues]
    payload["safe_fields_only"] = not issues
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect persisted LKW Elasticsearch failed-delivery JSONL records.",
    )
    parser.add_argument(
        "--file",
        required=True,
        type=Path,
        help="Path to failed-delivery JSONL file.",
    )
    parser.add_argument(
        "--check-safety",
        action="store_true",
        help="Exit non-zero when any JSONL object is not exactly the safe failed-delivery field set.",
    )
    parser.add_argument("--json", action="store_true", help="Output summary and records as JSON.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.file.exists():
        print(f"error: failed-delivery JSONL file does not exist: {args.file}", file=sys.stderr)
        print(
            "hint: enable LOCAL_WORKSPACE_OBSERVABILITY_ELASTICSEARCH_FAILED_DELIVERY_FILE_PATH "
            "and trigger a controlled Elasticsearch delivery failure first.",
            file=sys.stderr,
        )
        return 2

    records, issues = load_records(args.file)

    if args.json:
        print(json.dumps(to_json_payload(args.file, records, issues), indent=2, ensure_ascii=False))
    else:
        print_summary(args.file, records, issues)

    if args.check_safety and issues:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
