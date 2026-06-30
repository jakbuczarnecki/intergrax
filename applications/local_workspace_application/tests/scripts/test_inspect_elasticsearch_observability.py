from __future__ import annotations

import importlib.util
import io
import json
from pathlib import Path
from typing import Any


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
