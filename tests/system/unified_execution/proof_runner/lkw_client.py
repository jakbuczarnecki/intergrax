# © Artur Czarnecki. All rights reserved.

"""HTTP client for LKW production certification endpoints."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

from tests.system.unified_execution.proof_runner.contracts import (
    AgentInvocationSummary,
    ApplicationRunSummary,
    LkwEvidenceSlice,
    LkwRunResponse,
    ProofConfig,
    RuntimeEventSummary,
    RuntimeToolEventEntry,
    SearchSummaryDiagnostic,
)


class LkwClientError(RuntimeError):
    pass


class LkwClient:
    def __init__(self, config: ProofConfig) -> None:
        self._config = config

    def wait_until_ready(self) -> None:
        deadline = time.monotonic() + self._config.readiness_timeout_seconds
        readiness_url = f"{self._config.base_url.rstrip('/')}/v1/local_workspace/readiness"
        last_error = "readiness_unreachable"
        while time.monotonic() < deadline:
            try:
                payload = self._get_json(readiness_url)
                accepts = payload.get("accepts_new_work")
                if accepts is True:
                    return
                last_error = f"readiness_not_accepting:{payload!r}"
            except (LkwClientError, TimeoutError, OSError) as exc:
                last_error = str(exc)
            time.sleep(3.0)
        raise LkwClientError(last_error)

    def run_index(self, *, source_paths: list[str]) -> LkwRunResponse:
        body = {
            "tenant_id": self._config.tenant_id,
            "message": "index certification workspace",
            "capability": "local.workspace.index",
            "metadata": {
                "tenant_id": self._config.tenant_id,
                "workspace_id": self._config.workspace_id,
                "collection_id": self._config.collection_id,
                "source_paths": source_paths,
            },
        }
        return self._post_run(body)

    def run_search(
        self,
        *,
        message: str,
        metadata: dict[str, object] | None = None,
    ) -> LkwRunResponse:
        body = {
            "tenant_id": self._config.tenant_id,
            "message": message,
            "capability": self._config.capability,
            "metadata": {
                "tenant_id": self._config.tenant_id,
                "workspace_id": self._config.workspace_id,
                "collection_id": self._config.collection_id,
                "query": message,
                "top_k": 5,
            },
        }
        if metadata:
            meta = body["metadata"]
            if isinstance(meta, dict):
                meta.update(metadata)
        return self._post_run(body)

    def run_pipeline(
        self,
        *,
        message: str,
        metadata: dict[str, object],
        source_paths: list[str],
    ) -> LkwRunResponse:
        body = {
            "tenant_id": self._config.tenant_id,
            "message": message,
            "capability": "local.workspace.pipeline",
            "metadata": {
                "tenant_id": self._config.tenant_id,
                "workspace_id": self._config.workspace_id,
                "collection_id": self._config.collection_id,
                "query": message,
                "top_k": 5,
                "source_paths": source_paths,
                **metadata,
            },
        }
        return self._post_run(body)

    def run_synthesize(
        self,
        *,
        message: str,
        metadata: dict[str, object],
    ) -> LkwRunResponse:
        body = {
            "tenant_id": self._config.tenant_id,
            "message": message,
            "capability": "local.workspace.synthesize",
            "metadata": {
                "tenant_id": self._config.tenant_id,
                "workspace_id": self._config.workspace_id,
                "collection_id": self._config.collection_id,
                "query": message,
                "top_k": 5,
                **metadata,
            },
        }
        return self._post_run(body)

    def run_tool_selection_qualification(
        self,
        *,
        message: str,
        metadata: dict[str, object] | None = None,
    ) -> LkwRunResponse:
        body = {
            "tenant_id": self._config.tenant_id,
            "message": message,
            "capability": "local.workspace.tool_selection_qualification",
            "metadata": {
                "tenant_id": self._config.tenant_id,
                "workspace_id": self._config.workspace_id,
                "shadow_workspace": True,
            },
        }
        if metadata:
            meta = body["metadata"]
            if isinstance(meta, dict):
                meta.update(metadata)
        return self._post_run(body)

    def run_web_search_qualification(
        self,
        *,
        message: str,
        metadata: dict[str, object] | None = None,
    ) -> LkwRunResponse:
        body = {
            "tenant_id": self._config.tenant_id,
            "message": message,
            "capability": "local.workspace.web_search_qualification",
            "metadata": {
                "tenant_id": self._config.tenant_id,
            },
        }
        if metadata:
            meta = body["metadata"]
            if isinstance(meta, dict):
                meta.update(metadata)
        return self._post_run(body)

    def _post_run(self, body: dict[str, object]) -> LkwRunResponse:
        url = f"{self._config.base_url.rstrip('/')}/v1/local_workspace/run"
        status, payload = self._post_json(url, body)
        if status != 200:
            raise LkwClientError(f"run_http_status_{status}")
        return _parse_run_response(payload)

    def _get_json(self, url: str) -> dict[str, object]:
        request = urllib.request.Request(
            url,
            headers=self._headers(),
            method="GET",
        )
        try:
            with urllib.request.urlopen(
                request,
                timeout=self._config.request_timeout_seconds,
            ) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            raise LkwClientError(f"http_{exc.code}") from exc
        except urllib.error.URLError as exc:
            raise LkwClientError("url_error") from exc
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            raise LkwClientError("response_not_object")
        return parsed

    def _post_json(
        self,
        url: str,
        body: dict[str, object],
    ) -> tuple[int, dict[str, object]]:
        encoded = json.dumps(body).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=encoded,
            headers={
                **self._headers(),
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request,
                timeout=self._config.request_timeout_seconds,
            ) as response:
                status = int(response.status)
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            raise LkwClientError(f"http_{exc.code}") from exc
        except urllib.error.URLError as exc:
            raise LkwClientError("url_error") from exc
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            raise LkwClientError("response_not_object")
        return status, parsed

    def _headers(self) -> dict[str, str]:
        return {"X-API-Key": self._config.api_key}


def _parse_run_response(payload: dict[str, object]) -> LkwRunResponse:
    task_id = payload.get("task_id")
    run_id = payload.get("run_id")
    state = payload.get("state")
    if not isinstance(task_id, str) or not isinstance(run_id, str) or not isinstance(state, str):
        raise LkwClientError("missing_identity_fields")
    metadata = payload.get("metadata")
    metadata_map = metadata if isinstance(metadata, dict) else {}
    return LkwRunResponse(
        task_id=task_id,
        run_id=run_id,
        state=state,
        answer=_optional_str(payload.get("answer")),
        agent_id=_optional_str(payload.get("agent_id")),
        application_run_summary=_parse_app_summary(metadata_map),
        lkw_evidence=_parse_lkw_evidence(metadata_map),
        runtime_event_summary=_parse_runtime_event_summary(metadata_map),
    )


def _parse_app_summary(metadata: dict[str, object]) -> ApplicationRunSummary | None:
    raw = metadata.get("application_run_summary.v1")
    if not isinstance(raw, dict):
        return None
    schema_version = raw.get("schema_version")
    task_id = raw.get("task_id")
    terminal_status = raw.get("terminal_status")
    if not isinstance(schema_version, str):
        return None
    if not isinstance(task_id, str) or not isinstance(terminal_status, str):
        return None
    invocations: list[AgentInvocationSummary] = []
    raw_invocations = raw.get("agent_invocations")
    if isinstance(raw_invocations, list):
        for entry in raw_invocations:
            if not isinstance(entry, dict):
                continue
            agent_id = entry.get("agent_id")
            run_id = entry.get("run_id")
            if not isinstance(agent_id, str) or not isinstance(run_id, str):
                continue
            invocations.append(
                AgentInvocationSummary(
                    agent_id=agent_id,
                    run_id=run_id,
                    total_llm_tokens=_optional_int(entry.get("total_llm_tokens")) or 0,
                    total_tool_calls=_optional_int(entry.get("total_tool_calls")) or 0,
                )
            )
    return ApplicationRunSummary(
        schema_version=schema_version,
        task_id=task_id,
        terminal_status=str(terminal_status),
        total_llm_tokens=_optional_int(raw.get("total_llm_tokens")) or 0,
        agent_invocations=invocations,
    )


def _parse_lkw_evidence(metadata: dict[str, object]) -> LkwEvidenceSlice | None:
    raw = metadata.get("lkw_evidence.v1")
    if not isinstance(raw, dict):
        return None
    schema_version = raw.get("schema_version")
    if not isinstance(schema_version, str):
        return None
    diagnostics: dict[str, SearchSummaryDiagnostic] = {}
    raw_diag = raw.get("diagnostics")
    if isinstance(raw_diag, dict):
        for key, value in raw_diag.items():
            if not isinstance(key, str) or not isinstance(value, dict):
                continue
            diagnostics[key] = SearchSummaryDiagnostic(
                num_results=_optional_int(value.get("num_results")),
                evidence_count=_optional_int(value.get("evidence_count")),
                source_refs=_optional_str_list(value.get("source_refs")),
                reason=_optional_str(value.get("reason")),
            )
    return LkwEvidenceSlice(
        schema_version=schema_version,
        capability=_optional_str(raw.get("capability")),
        agent_id=_optional_str(raw.get("agent_id")),
        run_id=_optional_str(raw.get("run_id")),
        task_id=_optional_str(raw.get("task_id")),
        diagnostics=diagnostics,
    )


def _parse_runtime_event_summary(metadata: dict[str, object]) -> RuntimeEventSummary | None:
    raw = metadata.get("runtime_event_summary.v1")
    if not isinstance(raw, dict):
        return None
    schema_version = raw.get("schema_version")
    if not isinstance(schema_version, str):
        return None
    tool_events = raw.get("tool_events")
    total = 0
    tools: list[RuntimeToolEventEntry] = []
    if isinstance(tool_events, dict):
        raw_total = tool_events.get("total")
        if isinstance(raw_total, int):
            total = raw_total
        raw_tools = tool_events.get("tools")
        if isinstance(raw_tools, list):
            for entry in raw_tools:
                if not isinstance(entry, dict):
                    continue
                tool_id = entry.get("tool_id")
                if not isinstance(tool_id, str):
                    continue
                tools.append(
                    RuntimeToolEventEntry(
                        tool_id=tool_id,
                        requested=_optional_int(entry.get("requested")) or 0,
                        completed=_optional_int(entry.get("completed")) or 0,
                    )
                )
    return RuntimeEventSummary(
        schema_version=schema_version,
        tool_events_total=total,
        tools=tools,
    )


def _optional_str(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _optional_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def _optional_str_list(value: object) -> list[str] | None:
    if not isinstance(value, list):
        return None
    items = [item for item in value if isinstance(item, str)]
    return items
