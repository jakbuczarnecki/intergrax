#!/usr/bin/env python3
"""Inspect persisted Local Workspace OTLP JSONL log records.

This is a lightweight developer tool for the LKW Docker Compose OTLP proof path.
It reads OpenTelemetry Collector file-exporter JSONL output, flattens OTLP log
attributes, and prints a run timeline plus duplicate export checks.

The tool intentionally prints only observability metadata fields already present
in the OTLP records. It does not inspect source documents, prompts, chunks, or
other product content.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

Record = dict[str, Any]
DuplicateKey = tuple[str, str, str, str, str, str]

INTERGRAX_FIELDS = {
    "run_id": "intergrax.run_id",
    "task_id": "intergrax.task_id",
    "event_id": "intergrax.event_id",
    "event_type": "intergrax.event_type",
    "agent_id": "intergrax.agent_id",
    "tool_id": "intergrax.tool_id",
    "capability": "intergrax.capability",
    "status": "intergrax.status",
    "tenant_id": "intergrax.tenant_id",
    "workspace_id": "intergrax.workspace_id",
    "latency_ms": "intergrax.latency_ms",
    "schema_id": "intergrax.schema_id",
}

FILTER_FIELDS = {
    "event_type": "event_type",
    "agent_id": "agent_id",
    "tool_id": "tool_id",
    "capability": "capability",
    "tenant_id": "tenant_id",
    "workspace_id": "workspace_id",
}


def _otlp_value(value: Any) -> Any:
    """Return a Python scalar from an OTLP AnyValue-like object."""
    if not isinstance(value, dict):
        return value

    for key in ("stringValue", "intValue", "doubleValue", "boolValue"):
        if key in value:
            raw = value[key]
            if key == "intValue":
                try:
                    return int(raw)
                except (TypeError, ValueError):
                    return raw
            if key == "doubleValue":
                try:
                    return float(raw)
                except (TypeError, ValueError):
                    return raw
            return raw

    # Keep simple unsupported OTLP values visible but compact.
    if "arrayValue" in value:
        return value["arrayValue"]
    if "kvlistValue" in value:
        return value["kvlistValue"]
    if "bytesValue" in value:
        return value["bytesValue"]
    return None


def _flatten_attributes(attributes: Iterable[dict[str, Any]] | None) -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for attr in attributes or []:
        key = attr.get("key")
        if not key:
            continue
        flattened[str(key)] = _otlp_value(attr.get("value", {}))
    return flattened


def _body_value(body: Any) -> Any:
    if isinstance(body, dict):
        return _otlp_value(body)
    return body


def _parse_time_unix_nano(raw: Any) -> int | None:
    if raw in (None, ""):
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _time_iso(time_unix_nano: int | None) -> str:
    if time_unix_nano is None:
        return ""
    seconds = time_unix_nano / 1_000_000_000
    return datetime.fromtimestamp(seconds, tz=timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _iter_otlp_records(payload: dict[str, Any], *, line_no: int) -> Iterable[Record]:
    for resource_log in payload.get("resourceLogs", []) or []:
        resource = resource_log.get("resource", {}) or {}
        resource_attrs = _flatten_attributes(resource.get("attributes"))

        for scope_log in resource_log.get("scopeLogs", []) or []:
            scope = scope_log.get("scope", {}) or {}
            scope_attrs = _flatten_attributes(scope.get("attributes"))

            for log_record in scope_log.get("logRecords", []) or []:
                record_attrs = _flatten_attributes(log_record.get("attributes"))
                attrs: dict[str, Any] = {}
                attrs.update(resource_attrs)
                attrs.update(scope_attrs)
                attrs.update(record_attrs)

                time_unix_nano = _parse_time_unix_nano(log_record.get("timeUnixNano"))
                body = _body_value(log_record.get("body"))

                flattened: Record = {
                    "line_no": line_no,
                    "record_index": -1,
                    "time_unix_nano": time_unix_nano,
                    "time_iso": _time_iso(time_unix_nano),
                    "body": body if body is not None else "",
                    "severity_text": log_record.get("severityText", ""),
                    "attributes": attrs,
                }

                for target, source in INTERGRAX_FIELDS.items():
                    flattened[target] = attrs.get(source, "")

                if not flattened.get("event_type") and flattened.get("body"):
                    flattened["event_type"] = str(flattened["body"])

                yield flattened


def load_records(path: Path) -> list[Record]:
    """Load flattened OTLP log records from a JSONL file."""
    records: list[Record] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError as exc:
                print(f"warning: skipped malformed JSON line {line_no}: {exc}", file=sys.stderr)
                continue

            if not isinstance(payload, dict):
                print(f"warning: skipped non-object JSON line {line_no}", file=sys.stderr)
                continue

            for record in _iter_otlp_records(payload, line_no=line_no):
                record["record_index"] = len(records)
                records.append(record)

    return records


def find_latest_run_id(records: Sequence[Record]) -> str | None:
    latest_key: tuple[int, int] | None = None
    latest_run_id: str | None = None

    for record in records:
        run_id = str(record.get("run_id") or "")
        if not run_id:
            continue
        time_unix_nano = record.get("time_unix_nano")
        time_key = int(time_unix_nano) if isinstance(time_unix_nano, int) else -1
        record_index = int(record.get("record_index") or 0)
        key = (time_key, record_index)
        if latest_key is None or key > latest_key:
            latest_key = key
            latest_run_id = run_id

    return latest_run_id


def apply_filters(records: Sequence[Record], args: argparse.Namespace, *, include_run_id: bool = True) -> list[Record]:
    filtered = list(records)

    if include_run_id and args.run_id:
        filtered = [record for record in filtered if str(record.get("run_id") or "") == args.run_id]

    for arg_name, record_field in FILTER_FIELDS.items():
        expected = getattr(args, arg_name, None)
        if expected:
            filtered = [record for record in filtered if str(record.get(record_field) or "") == expected]

    return filtered


def duplicate_key(record: Record) -> DuplicateKey | None:
    event_id = str(record.get("event_id") or "")
    if not event_id:
        return None
    return (
        str(record.get("run_id") or ""),
        event_id,
        str(record.get("event_type") or ""),
        str(record.get("agent_id") or ""),
        str(record.get("tool_id") or ""),
        str(record.get("capability") or ""),
    )


def find_duplicate_groups(records: Sequence[Record]) -> list[tuple[DuplicateKey, int, list[Record]]]:
    keyed_records: defaultdict[DuplicateKey, list[Record]] = defaultdict(list)
    for record in records:
        key = duplicate_key(record)
        if key is not None:
            keyed_records[key].append(record)

    duplicates = [(key, len(rows), rows) for key, rows in keyed_records.items() if len(rows) > 1]
    duplicates.sort(key=lambda item: (item[0][0], item[0][1], item[0][2]))
    return duplicates


def select_records(records: Sequence[Record], args: argparse.Namespace) -> tuple[list[Record], str | None]:
    filtered = apply_filters(records, args, include_run_id=False)

    selected_run_id: str | None = None
    should_select_latest = args.latest_run

    default_mode = not any(
        [
            args.run_id,
            args.latest_run,
            args.list_runs,
            args.check_duplicates,
            args.json,
            args.event_type,
            args.agent_id,
            args.tool_id,
            args.capability,
            args.tenant_id,
            args.workspace_id,
        ]
    )
    if default_mode:
        should_select_latest = True

    if args.run_id:
        selected_run_id = args.run_id
        filtered = [record for record in filtered if str(record.get("run_id") or "") == selected_run_id]
    elif should_select_latest:
        selected_run_id = find_latest_run_id(filtered)
        if selected_run_id:
            filtered = [record for record in filtered if str(record.get("run_id") or "") == selected_run_id]

    return filtered, selected_run_id


def _truncate(value: Any, width: int) -> str:
    text = "" if value is None else str(value)
    if len(text) <= width:
        return text
    if width <= 1:
        return text[:width]
    return text[: width - 1] + "…"


def print_timeline(records: Sequence[Record], *, limit: int) -> None:
    rows = sorted(records, key=lambda record: ((record.get("time_unix_nano") or -1), record.get("record_index") or 0))
    if limit > 0:
        rows = rows[:limit]

    columns = [
        ("time_iso", "time", 24),
        ("event_type", "event_type", 28),
        ("agent_id", "agent", 18),
        ("tool_id", "tool", 18),
        ("capability", "capability", 28),
        ("status", "status", 10),
        ("latency_ms", "latency_ms", 10),
    ]
    header = " ".join(f"{title:<{width}}" for _field, title, width in columns)
    print(header)
    print(" ".join("-" * width for _field, _title, width in columns))
    for record in rows:
        print(" ".join(f"{_truncate(record.get(field, ''), width):<{width}}" for field, _title, width in columns))


def print_list_runs(records: Sequence[Record], *, limit: int) -> None:
    grouped: dict[str, list[Record]] = defaultdict(list)
    for record in records:
        run_id = str(record.get("run_id") or "")
        if run_id:
            grouped[run_id].append(record)

    summary = []
    for run_id, rows in grouped.items():
        latest = max((row.get("time_unix_nano") or -1 for row in rows), default=-1)
        latest_iso = _time_iso(latest if latest >= 0 else None)
        event_counts = Counter(str(row.get("event_type") or "") for row in rows)
        tenant_ids = sorted({str(row.get("tenant_id") or "") for row in rows if row.get("tenant_id")})
        workspace_ids = sorted({str(row.get("workspace_id") or "") for row in rows if row.get("workspace_id")})
        summary.append((latest, run_id, len(rows), latest_iso, tenant_ids, workspace_ids, event_counts))

    summary.sort(key=lambda item: (item[0], item[1]), reverse=True)
    if limit > 0:
        summary = summary[:limit]

    print(f"{'latest_time':<24} {'records':>7} {'run_id':<40} {'tenant':<18} {'workspace':<18} top_events")
    print(f"{'-' * 24} {'-' * 7} {'-' * 40} {'-' * 18} {'-' * 18} {'-' * 30}")
    for _latest, run_id, count, latest_iso, tenant_ids, workspace_ids, event_counts in summary:
        top_events = ", ".join(f"{name}:{qty}" for name, qty in event_counts.most_common(4) if name)
        print(
            f"{_truncate(latest_iso, 24):<24} {count:>7} {_truncate(run_id, 40):<40} "
            f"{_truncate(','.join(tenant_ids), 18):<18} {_truncate(','.join(workspace_ids), 18):<18} {top_events}"
        )


def print_duplicate_report(duplicates: Sequence[tuple[DuplicateKey, int, list[Record]]]) -> None:
    if not duplicates:
        print("Duplicate check: 0 duplicates")
        return

    print(f"Duplicate check: {len(duplicates)} duplicate group(s)")
    for key, count, rows in duplicates:
        run_id, event_id, event_type, agent_id, tool_id, capability = key
        lines = ",".join(str(row.get("line_no")) for row in rows)
        print(
            "- "
            f"count={count} run_id={run_id} event_id={event_id} event_type={event_type} "
            f"agent_id={agent_id} tool_id={tool_id} capability={capability} lines={lines}"
        )


def to_json_records(records: Sequence[Record], *, limit: int) -> list[dict[str, Any]]:
    rows = list(records)
    if limit > 0:
        rows = rows[:limit]
    fields = [
        "line_no",
        "record_index",
        "time_unix_nano",
        "time_iso",
        "body",
        "severity_text",
        "run_id",
        "task_id",
        "event_id",
        "event_type",
        "agent_id",
        "tool_id",
        "capability",
        "status",
        "tenant_id",
        "workspace_id",
        "latency_ms",
        "schema_id",
    ]
    return [{field: record.get(field, "") for field in fields} for record in rows]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect persisted LKW OTLP JSONL logs.")
    parser.add_argument("--file", required=True, type=Path, help="Path to OTLP JSONL file.")
    parser.add_argument("--run-id", help="Show records for a selected run id.")
    parser.add_argument("--latest-run", action="store_true", help="Show records for the latest run by timeUnixNano.")
    parser.add_argument("--list-runs", action="store_true", help="List discovered run ids with counts and latest event time.")
    parser.add_argument("--check-duplicates", action="store_true", help="Exit non-zero when duplicate event exports are found.")
    parser.add_argument("--json", action="store_true", help="Output selected records as JSON array.")
    parser.add_argument("--limit", type=int, default=100, help="Limit output rows. Use 0 for no limit. Default: 100.")
    parser.add_argument("--event-type", help="Filter by intergrax.event_type.")
    parser.add_argument("--agent-id", help="Filter by intergrax.agent_id.")
    parser.add_argument("--tool-id", help="Filter by intergrax.tool_id.")
    parser.add_argument("--capability", help="Filter by intergrax.capability.")
    parser.add_argument("--tenant-id", help="Filter by intergrax.tenant_id.")
    parser.add_argument("--workspace-id", help="Filter by intergrax.workspace_id.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.file.exists():
        print(f"error: OTLP JSONL file does not exist: {args.file}", file=sys.stderr)
        print("hint: run the LKW Docker Compose stack and execute a /v1/local_workspace/run request first.", file=sys.stderr)
        return 2

    records = load_records(args.file)
    selected, selected_run_id = select_records(records, args)

    if args.list_runs:
        list_records = apply_filters(records, args, include_run_id=True)
        print(f"File: {args.file}")
        print_list_runs(list_records, limit=args.limit)
        return 0

    duplicates = find_duplicate_groups(selected)

    if args.json:
        print(json.dumps(to_json_records(selected, limit=args.limit), indent=2, ensure_ascii=False))
    else:
        print(f"File: {args.file}")
        if selected_run_id:
            print(f"Run: {selected_run_id}")
        print(f"Records: {len(selected)}")
        print_duplicate_report(duplicates)
        if selected:
            print()
            print_timeline(selected, limit=args.limit)
        else:
            print("No records matched the selected filters.")

    if args.check_duplicates and duplicates:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
