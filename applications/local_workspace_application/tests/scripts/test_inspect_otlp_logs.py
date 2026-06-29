from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "inspect_otlp_logs.py"
SPEC = importlib.util.spec_from_file_location("inspect_otlp_logs", SCRIPT_PATH)
assert SPEC is not None
inspect_otlp_logs = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(inspect_otlp_logs)


def _attr(key: str, value: str | int | bool | float) -> dict[str, Any]:
    if isinstance(value, bool):
        otlp_value: dict[str, Any] = {"boolValue": value}
    elif isinstance(value, int):
        otlp_value = {"intValue": str(value)}
    elif isinstance(value, float):
        otlp_value = {"doubleValue": value}
    else:
        otlp_value = {"stringValue": value}
    return {"key": key, "value": otlp_value}


def _record(
    *,
    run_id: str,
    event_id: str,
    event_type: str,
    time_unix_nano: int,
    agent_id: str = "",
    tool_id: str = "",
    capability: str = "",
    latency_ms: int | None = None,
) -> dict[str, Any]:
    attrs = [
        _attr("intergrax.run_id", run_id),
        _attr("intergrax.task_id", run_id),
        _attr("intergrax.event_id", event_id),
        _attr("intergrax.event_type", event_type),
        _attr("intergrax.status", "unknown"),
    ]
    if agent_id:
        attrs.append(_attr("intergrax.agent_id", agent_id))
    if tool_id:
        attrs.append(_attr("intergrax.tool_id", tool_id))
    if capability:
        attrs.append(_attr("intergrax.capability", capability))
    if latency_ms is not None:
        attrs.append(_attr("intergrax.latency_ms", latency_ms))

    return {
        "timeUnixNano": str(time_unix_nano),
        "severityText": "UNKNOWN",
        "body": {"stringValue": event_type},
        "attributes": attrs,
    }


def _payload(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "resourceLogs": [
            {
                "resource": {
                    "attributes": [
                        _attr("service.name", "intergrax-lkw"),
                        _attr("service.version", "dev"),
                    ]
                },
                "scopeLogs": [
                    {
                        "scope": {"name": "intergrax.observability.export"},
                        "logRecords": records,
                    }
                ],
            }
        ]
    }


def _write_jsonl(path: Path, payloads: list[dict[str, Any]]) -> None:
    path.write_text("\n".join(json.dumps(payload) for payload in payloads), encoding="utf-8")


def test_parse_otlp_jsonl_extracts_log_records(tmp_path: Path) -> None:
    path = tmp_path / "logs.jsonl"
    _write_jsonl(
        path,
        [
            _payload(
                [
                    _record(
                        run_id="run-1",
                        event_id="evt-1",
                        event_type="tool_requested",
                        time_unix_nano=100,
                        agent_id="local_search",
                        tool_id="rag.retrieve",
                        capability="local.workspace.search",
                    ),
                    _record(
                        run_id="run-1",
                        event_id="evt-2",
                        event_type="tool_completed",
                        time_unix_nano=200,
                        agent_id="local_search",
                        tool_id="rag.retrieve",
                        capability="local.workspace.search",
                        latency_ms=42,
                    ),
                ]
            )
        ],
    )

    records = inspect_otlp_logs.load_records(path)

    assert len(records) == 2
    assert records[0]["run_id"] == "run-1"
    assert records[0]["tool_id"] == "rag.retrieve"
    assert records[1]["event_type"] == "tool_completed"
    assert records[1]["latency_ms"] == 42


def test_latest_run_selects_most_recent_run(tmp_path: Path) -> None:
    path = tmp_path / "logs.jsonl"
    _write_jsonl(
        path,
        [
            _payload([_record(run_id="run-old", event_id="evt-old", event_type="task_completed", time_unix_nano=100)]),
            _payload([_record(run_id="run-new", event_id="evt-new", event_type="task_completed", time_unix_nano=300)]),
        ],
    )

    records = inspect_otlp_logs.load_records(path)

    assert inspect_otlp_logs.find_latest_run_id(records) == "run-new"


def test_duplicate_check_detects_same_event_id_key(tmp_path: Path) -> None:
    path = tmp_path / "logs.jsonl"
    duplicated = _record(
        run_id="run-1",
        event_id="evt-dup",
        event_type="tool_requested",
        time_unix_nano=100,
        agent_id="local_search",
        tool_id="rag.retrieve",
        capability="local.workspace.search",
    )
    _write_jsonl(path, [_payload([duplicated, duplicated])])

    records = inspect_otlp_logs.load_records(path)
    duplicates = inspect_otlp_logs.find_duplicate_groups(records)

    assert len(duplicates) == 1
    assert duplicates[0][1] == 2
    assert duplicates[0][0][1] == "evt-dup"


def test_duplicate_check_allows_different_event_ids(tmp_path: Path) -> None:
    path = tmp_path / "logs.jsonl"
    _write_jsonl(
        path,
        [
            _payload(
                [
                    _record(
                        run_id="run-1",
                        event_id="evt-1",
                        event_type="tool_requested",
                        time_unix_nano=100,
                        agent_id="local_search",
                        tool_id="rag.retrieve",
                        capability="local.workspace.search",
                    ),
                    _record(
                        run_id="run-1",
                        event_id="evt-2",
                        event_type="tool_requested",
                        time_unix_nano=200,
                        agent_id="local_search",
                        tool_id="rag.retrieve",
                        capability="local.workspace.search",
                    ),
                ]
            )
        ],
    )

    records = inspect_otlp_logs.load_records(path)

    assert inspect_otlp_logs.find_duplicate_groups(records) == []


def test_filters_by_tool_id_and_run_id(tmp_path: Path) -> None:
    path = tmp_path / "logs.jsonl"
    _write_jsonl(
        path,
        [
            _payload(
                [
                    _record(
                        run_id="run-1",
                        event_id="evt-1",
                        event_type="tool_requested",
                        time_unix_nano=100,
                        agent_id="local_search",
                        tool_id="rag.retrieve",
                    ),
                    _record(
                        run_id="run-2",
                        event_id="evt-2",
                        event_type="tool_requested",
                        time_unix_nano=200,
                        agent_id="local_indexer",
                        tool_id="rag.ingest_document",
                    ),
                ]
            )
        ],
    )

    records = inspect_otlp_logs.load_records(path)
    args = inspect_otlp_logs.build_parser().parse_args(
        [
            "--file",
            str(path),
            "--run-id",
            "run-1",
            "--tool-id",
            "rag.retrieve",
        ]
    )

    filtered, selected_run_id = inspect_otlp_logs.select_records(records, args)

    assert selected_run_id == "run-1"
    assert len(filtered) == 1
    assert filtered[0]["tool_id"] == "rag.retrieve"
