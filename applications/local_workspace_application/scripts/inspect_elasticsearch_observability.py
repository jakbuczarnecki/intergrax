#!/usr/bin/env python3
"""Inspect persisted Local Workspace observability documents in Elasticsearch/OpenSearch.

Lightweight readback tooling for the LKW Docker Compose Elasticsearch proof path.
Queries policy-safe indexed documents only; does not modify the index or runtime export.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from typing import Any, Callable, Iterable, Sequence

Record = dict[str, Any]
DuplicateKey = tuple[str, str, str, str, str, str]
SafetyViolation = tuple[str, list[str]]

DEFAULT_URL = "http://127.0.0.1:9200"
DEFAULT_INDEX = "intergrax-lkw-observability"

INTERGRAX_SOURCE_FIELDS = (
    "intergrax.run_id",
    "intergrax.event_id",
    "intergrax.event_type",
    "intergrax.agent_id",
    "intergrax.tool_id",
    "intergrax.capability",
    "intergrax.status",
)

RECORD_FIELDS = (
    "run_id",
    "event_id",
    "event_type",
    "agent_id",
    "tool_id",
    "capability",
    "status",
    "timestamp",
    "doc_id",
)

FORBIDDEN_KEY_FRAGMENTS = (
    "prompt",
    "completion",
    "content",
    "chunk",
    "chunks",
    "query",
    "tool_args",
    "secret",
    "token",
    "password",
    "absolute_path",
    "full_path",
)

UrlOpener = Callable[[urllib.request.Request], Any]


def normalize_url(url: str) -> str:
    return url.rstrip("/")


def build_search_path(index: str) -> str:
    return f"/{index}/_search"


def build_search_url(base_url: str, index: str) -> str:
    return f"{normalize_url(base_url)}{build_search_path(index)}"


def build_list_runs_query(*, limit: int) -> dict[str, Any]:
    return {
        "size": limit,
        "sort": [{"@timestamp": {"order": "desc", "unmapped_type": "date"}}],
        "query": {"match_all": {}},
    }


def build_run_id_query(run_id: str, *, limit: int) -> dict[str, Any]:
    return {
        "size": limit,
        "sort": [{"@timestamp": {"order": "asc", "unmapped_type": "date"}}],
        "query": {
            "bool": {
                "should": [
                    {"term": {"intergrax.run_id.keyword": run_id}},
                    {"term": {"intergrax.run_id": run_id}},
                ],
                "minimum_should_match": 1,
            }
        },
    }


def _get_nested_value(source: dict[str, Any], dotted_key: str) -> Any:
    if dotted_key in source:
        return source[dotted_key]

    if dotted_key.startswith("intergrax."):
        intergrax = source.get("intergrax")
        if isinstance(intergrax, dict):
            short_key = dotted_key.removeprefix("intergrax.")
            if short_key in intergrax:
                return intergrax[short_key]

    return ""


def extract_record(hit: dict[str, Any]) -> Record:
    source = hit.get("_source", {})
    if not isinstance(source, dict):
        source = {}

    record: Record = {
        "doc_id": str(hit.get("_id", "")),
        "timestamp": str(source.get("@timestamp", "")),
        "source": source,
    }
    for field in RECORD_FIELDS:
        if field in record:
            continue
        source_key = f"intergrax.{field}" if field != "timestamp" else "@timestamp"
        if field == "timestamp":
            record[field] = str(source.get("@timestamp", ""))
        else:
            value = _get_nested_value(source, source_key)
            record[field] = "" if value is None else str(value)
    return record


def parse_hits(response: dict[str, Any]) -> list[Record]:
    hits = response.get("hits", {}).get("hits", [])
    if not isinstance(hits, list):
        return []
    return [extract_record(hit) for hit in hits if isinstance(hit, dict)]


def summarize_runs(records: Sequence[Record]) -> list[tuple[str, int, str]]:
    grouped: dict[str, list[Record]] = defaultdict(list)
    for record in records:
        run_id = str(record.get("run_id") or "")
        if run_id:
            grouped[run_id].append(record)

    summary: list[tuple[str, int, str]] = []
    for run_id, rows in grouped.items():
        timestamps = [str(row.get("timestamp") or "") for row in rows if row.get("timestamp")]
        latest = max(timestamps) if timestamps else ""
        summary.append((run_id, len(rows), latest))

    summary.sort(key=lambda item: (item[2], item[0]), reverse=True)
    return summary


def sort_timeline(records: Sequence[Record]) -> list[Record]:
    return sorted(records, key=lambda record: (str(record.get("timestamp") or ""), str(record.get("doc_id") or "")))


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


def _key_name(key: Any) -> str:
    return str(key)


def key_has_forbidden_fragment(key: str) -> bool:
    lowered = key.lower()
    return any(fragment in lowered for fragment in FORBIDDEN_KEY_FRAGMENTS)


def find_forbidden_keys(value: Any, *, path: str = "") -> list[str]:
    offenders: list[str] = []

    if isinstance(value, dict):
        for key, nested in value.items():
            key_name = _key_name(key)
            child_path = f"{path}.{key_name}" if path else key_name
            if key_has_forbidden_fragment(key_name):
                offenders.append(child_path)
            offenders.extend(find_forbidden_keys(nested, path=child_path))
        return offenders

    if isinstance(value, list):
        for index, item in enumerate(value):
            child_path = f"{path}[{index}]"
            offenders.extend(find_forbidden_keys(item, path=child_path))
        return offenders

    return offenders


def check_safety(records: Sequence[Record]) -> list[SafetyViolation]:
    violations: list[SafetyViolation] = []
    for record in records:
        source = record.get("source")
        if not isinstance(source, dict):
            continue
        offenders = find_forbidden_keys(source)
        if offenders:
            doc_id = str(record.get("doc_id") or "")
            violations.append((doc_id, offenders))
    return violations


def elasticsearch_search(
    *,
    url: str,
    index: str,
    body: dict[str, Any],
    opener: UrlOpener | None = None,
) -> dict[str, Any]:
    search_url = build_search_url(url, index)
    payload = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        search_url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    open_request = opener or urllib.request.urlopen
    try:
        with open_request(request) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Elasticsearch search failed ({exc.code}): {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Elasticsearch search failed: {exc.reason}") from exc

    parsed = json.loads(raw.decode("utf-8"))
    if not isinstance(parsed, dict):
        raise RuntimeError("Elasticsearch search returned a non-object JSON response.")
    return parsed


def _truncate(value: Any, width: int) -> str:
    text = "" if value is None else str(value)
    if len(text) <= width:
        return text
    if width <= 1:
        return text[:width]
    return text[: width - 1] + "…"


def print_list_runs(summary: Sequence[tuple[str, int, str]], *, url: str, index: str) -> None:
    print(f"URL: {url}")
    print(f"Index: {index}")
    print(f"{'latest_timestamp':<28} {'count':>7} run_id")
    print(f"{'-' * 28} {'-' * 7} {'-' * 40}")
    for run_id, count, latest in summary:
        print(f"{_truncate(latest, 28):<28} {count:>7} {run_id}")


def print_timeline(records: Sequence[Record], *, run_id: str, url: str, index: str) -> None:
    print(f"URL: {url}")
    print(f"Index: {index}")
    print(f"Run: {run_id}")
    print(f"Records: {len(records)}")
    columns = [
        ("timestamp", "timestamp", 28),
        ("event_type", "event_type", 24),
        ("agent_id", "agent_id", 16),
        ("tool_id", "tool_id", 18),
        ("capability", "capability", 24),
        ("status", "status", 10),
        ("event_id", "event_id", 24),
    ]
    header = " ".join(f"{title:<{width}}" for _field, title, width in columns)
    print(header)
    print(" ".join("-" * width for _field, _title, width in columns))
    for record in records:
        print(
            " ".join(
                f"{_truncate(record.get(field, ''), width):<{width}}"
                for field, _title, width in columns
            )
        )


def print_duplicate_report(duplicates: Sequence[tuple[DuplicateKey, int, list[Record]]]) -> None:
    if not duplicates:
        print("Duplicate check: duplicate groups = 0")
        return

    print(f"Duplicate check: duplicate groups = {len(duplicates)}")
    for key, count, rows in duplicates:
        run_id, event_id, event_type, agent_id, tool_id, capability = key
        doc_ids = ",".join(str(row.get("doc_id") or "") for row in rows)
        print(
            "- "
            f"count={count} run_id={run_id} event_id={event_id} event_type={event_type} "
            f"agent_id={agent_id} tool_id={tool_id} capability={capability} doc_ids={doc_ids}"
        )


def print_safety_report(violations: Sequence[SafetyViolation]) -> None:
    if not violations:
        print("Safety check: 0 forbidden keys")
        return

    print(f"Safety check: {len(violations)} document(s) with forbidden keys")
    for doc_id, keys in violations:
        unique_keys = sorted(set(keys))
        print(f"- doc_id={doc_id} keys={', '.join(unique_keys)}")


def to_json_records(records: Sequence[Record]) -> list[dict[str, Any]]:
    return [{field: record.get(field, "") for field in RECORD_FIELDS} for record in records]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect persisted LKW observability documents in Elasticsearch/OpenSearch.",
    )
    parser.add_argument("--url", default=DEFAULT_URL, help=f"Elasticsearch base URL. Default: {DEFAULT_URL}")
    parser.add_argument("--index", default=DEFAULT_INDEX, help=f"Index name. Default: {DEFAULT_INDEX}")
    parser.add_argument("--list-runs", action="store_true", help="List distinct run ids with counts.")
    parser.add_argument("--run-id", help="Inspect a selected run id.")
    parser.add_argument("--limit", type=int, default=100, help="Maximum documents to fetch. Default: 100.")
    parser.add_argument(
        "--check-duplicates",
        action="store_true",
        help="Exit non-zero when duplicate event exports are found for the selected run.",
    )
    parser.add_argument(
        "--check-safety",
        action="store_true",
        help="Exit non-zero when forbidden raw-content key fragments are present.",
    )
    parser.add_argument("--json", action="store_true", help="Output selected records as JSON.")
    return parser


def main(argv: Sequence[str] | None = None, *, opener: UrlOpener | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.limit < 0:
        print("error: --limit must be >= 0", file=sys.stderr)
        return 2

    if args.list_runs:
        response = elasticsearch_search(
            url=args.url,
            index=args.index,
            body=build_list_runs_query(limit=args.limit),
            opener=opener,
        )
        records = parse_hits(response)
        summary = summarize_runs(records)
        if args.json:
            print(
                json.dumps(
                    [{"run_id": run_id, "count": count, "latest_timestamp": latest} for run_id, count, latest in summary],
                    indent=2,
                    ensure_ascii=False,
                )
            )
        else:
            print_list_runs(summary, url=args.url, index=args.index)
        return 0

    if not args.run_id:
        print("error: --run-id is required unless --list-runs is used", file=sys.stderr)
        return 2

    response = elasticsearch_search(
        url=args.url,
        index=args.index,
        body=build_run_id_query(args.run_id, limit=args.limit),
        opener=opener,
    )
    records = sort_timeline(parse_hits(response))
    duplicates = find_duplicate_groups(records)
    violations = check_safety(records)

    if args.json:
        payload: dict[str, Any] = {"records": to_json_records(records)}
        if args.check_duplicates:
            payload["duplicate_groups"] = len(duplicates)
        if args.check_safety:
            payload["safety_violations"] = [
                {"doc_id": doc_id, "keys": keys} for doc_id, keys in violations
            ]
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print_timeline(records, run_id=args.run_id, url=args.url, index=args.index)
        if args.check_duplicates:
            print()
            print_duplicate_report(duplicates)
        if args.check_safety:
            print()
            print_safety_report(violations)

    exit_code = 0
    if args.check_duplicates and duplicates:
        exit_code = 1
    if args.check_safety and violations:
        exit_code = 1
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
