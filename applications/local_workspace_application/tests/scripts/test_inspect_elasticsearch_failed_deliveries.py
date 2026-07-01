from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

from dataclasses import fields

from intergrax.integrations.providers.observability_backend.elasticsearch.transport import (
    ElasticsearchFailedDeliveryRecord,
)


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "inspect_elasticsearch_failed_deliveries.py"
)
SPEC = importlib.util.spec_from_file_location("inspect_elasticsearch_failed_deliveries", SCRIPT_PATH)
assert SPEC is not None
inspect_failed = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(inspect_failed)

_SAFE_FIELD_NAMES = frozenset(field.name for field in fields(ElasticsearchFailedDeliveryRecord))


def _sample_record(**overrides: object) -> dict[str, Any]:
    defaults: dict[str, Any] = {
        "provider_id": "elasticsearch",
        "operation": "send_observability_payload",
        "index": "intergrax-lkw-observability",
        "status_code": 503,
        "reason": "http_status_503",
        "retriable": True,
        "attempts": 3,
        "exhausted": True,
    }
    defaults.update(overrides)
    return defaults


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")


def test_validate_record_object_accepts_safe_fields_only() -> None:
    issues = inspect_failed.validate_record_object(_sample_record(), line_no=1)
    assert issues == []


def test_validate_record_object_rejects_extra_keys() -> None:
    payload = _sample_record(prompt="RAW_PROMPT_DO_NOT_LEAK")
    issues = inspect_failed.validate_record_object(payload, line_no=2)
    assert issues == [(2, "unexpected keys: prompt")]


def test_validate_record_object_rejects_missing_keys() -> None:
    payload = _sample_record()
    payload.pop("reason")
    issues = inspect_failed.validate_record_object(payload, line_no=3)
    assert issues == [(3, "missing keys: reason")]


def test_load_records_parses_valid_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "failed-deliveries.jsonl"
    _write_jsonl(
        path,
        [
            _sample_record(status_code=503, reason="http_status_503"),
            _sample_record(status_code=None, reason="connection_error", retriable=False, attempts=1, exhausted=False),
        ],
    )

    records, issues = inspect_failed.load_records(path)

    assert issues == []
    assert len(records) == 2
    assert records[0]["line_no"] == 1
    assert records[1]["status_code"] is None
    assert set(records[0].keys()) == _SAFE_FIELD_NAMES | {"line_no"}


def test_summarize_records_counts_reason_and_status() -> None:
    records = [
        {"reason": "http_status_503", "retriable": True, "exhausted": True, "status_code": 503},
        {"reason": "http_status_503", "retriable": False, "exhausted": False, "status_code": 400},
    ]

    summary = inspect_failed.summarize_records(records)

    assert summary["record_count"] == 2
    assert summary["retriable_count"] == 1
    assert summary["exhausted_count"] == 1
    assert summary["reason_counts"] == {"http_status_503": 2}
    assert summary["status_code_counts"] == {"503": 1, "400": 1}


def test_main_check_safety_passes_for_valid_file(tmp_path: Path, capsys: Any) -> None:
    path = tmp_path / "failed-deliveries.jsonl"
    _write_jsonl(path, [_sample_record()])

    exit_code = inspect_failed.main(["--file", str(path), "--check-safety"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Validation: all records contain exactly the safe failed-delivery fields" in captured.out


def test_main_check_safety_fails_for_unsafe_keys(tmp_path: Path, capsys: Any) -> None:
    path = tmp_path / "failed-deliveries.jsonl"
    _write_jsonl(path, [_sample_record(document="must-not-appear")])

    exit_code = inspect_failed.main(["--file", str(path), "--check-safety"])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "unexpected keys: document" in captured.out


def test_main_json_output_includes_safe_fields_only(tmp_path: Path, capsys: Any) -> None:
    path = tmp_path / "failed-deliveries.jsonl"
    _write_jsonl(path, [_sample_record()])

    exit_code = inspect_failed.main(["--file", str(path), "--json"])

    captured = capsys.readouterr()
    assert exit_code == 0
    payload = json.loads(captured.out)
    assert payload["record_count"] == 1
    assert payload["safe_fields_only"] is True
    assert set(payload["records"][0].keys()) == _SAFE_FIELD_NAMES


def test_main_missing_file_returns_error(tmp_path: Path, capsys: Any) -> None:
    path = tmp_path / "missing.jsonl"

    exit_code = inspect_failed.main(["--file", str(path)])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "does not exist" in captured.err
