from __future__ import annotations

import importlib.util
import io
import json
import urllib.error
from pathlib import Path
from typing import Any

from intergrax.runtime.observability.export_boundary import FORBIDDEN_EXPORT_CONTENT_FIELDS


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "inspect_elasticsearch_observability.py"
SPEC = importlib.util.spec_from_file_location("inspect_elasticsearch_observability", SCRIPT_PATH)
assert SPEC is not None
inspect_es = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(inspect_es)


def _hit(
    *,
    doc_id: str,
    run_id: str,
    event_id: str,
    event_type: str,
    timestamp: str,
    agent_id: str = "local_search",
    tool_id: str = "rag.retrieve",
    capability: str = "local.workspace.search",
    status: str = "unknown",
    extra_source: dict[str, Any] | None = None,
) -> dict[str, Any]:
    source: dict[str, Any] = {
        "@timestamp": timestamp,
        "intergrax.run_id": run_id,
        "intergrax.event_id": event_id,
        "intergrax.event_type": event_type,
        "intergrax.agent_id": agent_id,
        "intergrax.tool_id": tool_id,
        "intergrax.capability": capability,
        "intergrax.status": status,
    }
    if extra_source:
        source.update(extra_source)
    return {"_id": doc_id, "_source": source}


def _search_response(hits: list[dict[str, Any]]) -> dict[str, Any]:
    return {"hits": {"hits": hits}}


def _fake_opener(response: dict[str, Any]):
    payload = json.dumps(response).encode("utf-8")

    def opener(request: Any) -> io.BytesIO:
        return io.BytesIO(payload)

    return opener


def _index_not_found_opener(request: Any) -> None:
    payload = {
        "error": {
            "root_cause": [
                {
                    "type": "index_not_found_exception",
                    "reason": "no such index [intergrax-lkw-observability]",
                    "index": "intergrax-lkw-observability",
                }
            ],
            "type": "index_not_found_exception",
            "reason": "no such index [intergrax-lkw-observability]",
            "index": "intergrax-lkw-observability",
        },
        "status": 404,
    }
    raise urllib.error.HTTPError(
        request.full_url,
        404,
        "Not Found",
        hdrs=None,
        fp=io.BytesIO(json.dumps(payload).encode("utf-8")),
    )


def test_build_search_url_uses_index_search_path() -> None:
    assert (
        inspect_es.build_search_url("http://127.0.0.1:9200", "intergrax-lkw-observability")
        == "http://127.0.0.1:9200/intergrax-lkw-observability/_search"
    )
    assert (
        inspect_es.build_search_url("http://127.0.0.1:9200/", "intergrax-lkw-observability")
        == "http://127.0.0.1:9200/intergrax-lkw-observability/_search"
    )


def test_list_runs_groups_run_ids_and_counts() -> None:
    hits = [
        _hit(doc_id="1", run_id="run-a", event_id="evt-1", event_type="tool_requested", timestamp="2026-06-30T10:00:00Z"),
        _hit(doc_id="2", run_id="run-a", event_id="evt-2", event_type="tool_completed", timestamp="2026-06-30T10:00:01Z"),
        _hit(doc_id="3", run_id="run-b", event_id="evt-3", event_type="task_completed", timestamp="2026-06-30T11:00:00Z"),
    ]
    records = inspect_es.parse_hits(_search_response(hits))
    summary = inspect_es.summarize_runs(records)

    assert summary == [
        ("run-b", 1, "2026-06-30T11:00:00Z"),
        ("run-a", 2, "2026-06-30T10:00:01Z"),
    ]


def test_list_runs_returns_success_when_index_is_missing(capsys: Any) -> None:
    exit_code = inspect_es.main(["--list-runs"], opener=_index_not_found_opener)

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "No observability index found yet" in output
    assert "before the first LKW run" in output


def test_list_runs_json_returns_empty_array_when_index_is_missing(capsys: Any) -> None:
    exit_code = inspect_es.main(["--list-runs", "--json"], opener=_index_not_found_opener)

    assert exit_code == 0
    assert json.loads(capsys.readouterr().out) == []


def test_run_id_returns_nonzero_when_index_is_missing(capsys: Any) -> None:
    exit_code = inspect_es.main(["--run-id", "run-missing"], opener=_index_not_found_opener)

    assert exit_code == 1
    error_output = capsys.readouterr().err
    assert "index not found" in error_output
    assert "execute a real LKW run" in error_output


def test_run_id_timeline_extracts_intergrax_fields() -> None:
    hits = [
        _hit(
            doc_id="doc-2",
            run_id="run-1",
            event_id="evt-2",
            event_type="tool_completed",
            timestamp="2026-06-30T10:00:02Z",
            status="ok",
        ),
        _hit(
            doc_id="doc-1",
            run_id="run-1",
            event_id="evt-1",
            event_type="tool_requested",
            timestamp="2026-06-30T10:00:01Z",
            status="started",
        ),
    ]
    records = inspect_es.sort_timeline(inspect_es.parse_hits(_search_response(hits)))

    assert [record["event_id"] for record in records] == ["evt-1", "evt-2"]
    assert records[0]["agent_id"] == "local_search"
    assert records[0]["tool_id"] == "rag.retrieve"
    assert records[0]["capability"] == "local.workspace.search"
    assert records[1]["status"] == "ok"


def test_duplicate_check_returns_success_when_no_duplicates() -> None:
    hits = [
        _hit(doc_id="1", run_id="run-1", event_id="evt-1", event_type="tool_requested", timestamp="t1"),
        _hit(doc_id="2", run_id="run-1", event_id="evt-2", event_type="tool_completed", timestamp="t2"),
    ]
    records = inspect_es.parse_hits(_search_response(hits))

    assert inspect_es.find_duplicate_groups(records) == []


def test_duplicate_check_exits_nonzero_when_duplicates_exist() -> None:
    duplicated = _hit(
        doc_id="1",
        run_id="run-1",
        event_id="evt-dup",
        event_type="tool_requested",
        timestamp="t1",
    )
    hits = [duplicated, {**duplicated, "_id": "2"}]
    records = inspect_es.parse_hits(_search_response(hits))
    duplicates = inspect_es.find_duplicate_groups(records)

    assert len(duplicates) == 1
    assert duplicates[0][1] == 2

    exit_code = inspect_es.main(
        ["--run-id", "run-1", "--check-duplicates"],
        opener=_fake_opener(_search_response(hits)),
    )

    assert exit_code == 1


def test_safety_check_uses_canonical_forbidden_export_fields() -> None:
    assert inspect_es.CANONICAL_FORBIDDEN_EXPORT_KEYS == FORBIDDEN_EXPORT_CONTENT_FIELDS
    assert inspect_es.CANONICAL_FORBIDDEN_EXPORT_KEYS == inspect_es.FORBIDDEN_EXPORT_CONTENT_FIELDS


def test_safety_check_fails_for_exact_forbidden_keys() -> None:
    for forbidden_key in ("prompt", "completion", "tool_args", "api_key"):
        hits = [
            _hit(
                doc_id=f"unsafe-{forbidden_key}",
                run_id="run-1",
                event_id=f"evt-{forbidden_key}",
                event_type="tool_requested",
                timestamp="t1",
                extra_source={forbidden_key: "leak"},
            ),
        ]
        records = inspect_es.parse_hits(_search_response(hits))
        violations = inspect_es.check_safety(records)

        assert violations, forbidden_key
        assert violations[0][0] == f"unsafe-{forbidden_key}"
        assert any(forbidden_key in key for key in violations[0][1])


def test_safety_check_fails_for_compound_forbidden_keys() -> None:
    compound_cases = {
        "raw_prompt": "leak",
        "user_prompt": "leak",
        "raw_chunks": [],
        "raw_content": "leak",
    }
    for compound_key, value in compound_cases.items():
        hits = [
            _hit(
                doc_id=f"unsafe-{compound_key}",
                run_id="run-1",
                event_id=f"evt-{compound_key}",
                event_type="tool_requested",
                timestamp="t1",
                extra_source={compound_key: value},
            ),
        ]
        records = inspect_es.parse_hits(_search_response(hits))
        violations = inspect_es.check_safety(records)

        assert violations, compound_key
        assert any(compound_key in key for key in violations[0][1])


def test_safety_check_fails_for_nested_forbidden_keys() -> None:
    hits = [
        _hit(
            doc_id="unsafe-nested",
            run_id="run-1",
            event_id="evt-nested",
            event_type="tool_requested",
            timestamp="t1",
            extra_source={"intergrax": {"prompt": "secret text"}},
        ),
    ]
    records = inspect_es.parse_hits(_search_response(hits))
    violations = inspect_es.check_safety(records)

    assert violations
    assert violations[0][0] == "unsafe-nested"
    assert any("prompt" in key for key in violations[0][1])


def test_safety_check_fails_for_list_nested_forbidden_keys() -> None:
    hits = [
        _hit(
            doc_id="unsafe-list",
            run_id="run-1",
            event_id="evt-list",
            event_type="tool_requested",
            timestamp="t1",
            extra_source={"items": [{"messages": ["leak"]}]},
        ),
    ]
    records = inspect_es.parse_hits(_search_response(hits))
    violations = inspect_es.check_safety(records)

    assert violations
    assert any("messages" in key for key in violations[0][1])


def test_safety_check_passes_for_policy_safe_intergrax_fields() -> None:
    safe_source = {
        "intergrax.status": "ok",
        "intergrax.run_id": "run-1",
        "intergrax.event_id": "evt-safe",
        "intergrax.tool_id": "rag.retrieve",
        "intergrax.capability": "local.workspace.search",
        "intergrax.safe_relative_path": "docs/README.md",
        "intergrax.sha256": "abc123",
        "intergrax.artifact_ref": "artifact://ref",
        "@timestamp": "2026-06-30T10:00:00Z",
    }
    hits = [
        _hit(
            doc_id="safe-1",
            run_id="run-1",
            event_id="evt-safe",
            event_type="tool_requested",
            timestamp="2026-06-30T10:00:00Z",
            extra_source=safe_source,
        ),
    ]
    records = inspect_es.parse_hits(_search_response(hits))

    assert inspect_es.check_safety(records) == []


def test_safety_check_allows_safe_relative_path_despite_path_substring() -> None:
    hits = [
        _hit(
            doc_id="safe-path",
            run_id="run-1",
            event_id="evt-path",
            event_type="tool_requested",
            timestamp="t1",
            extra_source={
                "intergrax.safe_relative_path": "docs/README.md",
                "intergrax": {"safe_relative_path": "docs/README.md"},
            },
        ),
    ]
    records = inspect_es.parse_hits(_search_response(hits))

    assert inspect_es.check_safety(records) == []
    assert not inspect_es.key_is_forbidden_export_field("safe_relative_path")
    assert not inspect_es.key_is_forbidden_export_field("intergrax.safe_relative_path")


def test_safety_check_passes_for_normal_intergrax_metadata() -> None:
    hits = [
        _hit(doc_id="1", run_id="run-1", event_id="evt-1", event_type="tool_requested", timestamp="t1"),
    ]
    records = inspect_es.parse_hits(_search_response(hits))

    assert inspect_es.check_safety(records) == []


def test_safety_check_fails_for_forbidden_raw_content_keys() -> None:
    hits = [
        _hit(
            doc_id="unsafe-1",
            run_id="run-1",
            event_id="evt-1",
            event_type="tool_requested",
            timestamp="t1",
            extra_source={"intergrax": {"prompt": "secret text"}},
        ),
    ]
    records = inspect_es.parse_hits(_search_response(hits))
    violations = inspect_es.check_safety(records)

    assert violations
    assert violations[0][0] == "unsafe-1"
    assert any("prompt" in key for key in violations[0][1])

    exit_code = inspect_es.main(
        ["--run-id", "run-1", "--check-safety"],
        opener=_fake_opener(_search_response(hits)),
    )

    assert exit_code == 1


def test_elasticsearch_search_uses_expected_request_path() -> None:
    captured: dict[str, Any] = {}

    class FakeResponse:
        def __init__(self, payload: bytes) -> None:
            self._payload = payload

        def read(self) -> bytes:
            return self._payload

        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

    def opener(request: Any) -> FakeResponse:
        captured["url"] = request.full_url
        captured["method"] = request.method
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return FakeResponse(json.dumps(_search_response([])).encode("utf-8"))

    inspect_es.elasticsearch_search(
        url="http://127.0.0.1:9200",
        index="intergrax-lkw-observability",
        body=inspect_es.build_run_id_query("run-1", limit=10),
        opener=opener,
    )

    assert captured["url"] == "http://127.0.0.1:9200/intergrax-lkw-observability/_search"
    assert captured["method"] == "POST"
    assert captured["body"]["query"]["bool"]["should"] == [
        {"term": {"intergrax.run_id.keyword": "run-1"}},
        {"term": {"intergrax.run_id": "run-1"}},
    ]
