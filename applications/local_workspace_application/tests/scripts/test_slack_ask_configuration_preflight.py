# © Artur Czarnecki. All rights reserved.

"""Tests for LKW Slack Ask configuration preflight (no live Slack/host)."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from collections.abc import Generator
from pathlib import Path
from typing import Any

import httpx
import pytest

from local_workspace_application.slack_companion.companion import (
    SLACK_COMPANION_PRODUCT_ENV_KEYS,
)

SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "run-lkw-slack-ask-configuration-preflight.py"
)
ENV_EXAMPLE_PATH = Path(__file__).resolve().parents[2] / ".env.example"

SPEC = importlib.util.spec_from_file_location(
    "run_lkw_slack_ask_configuration_preflight",
    SCRIPT_PATH,
)
assert SPEC is not None and SPEC.loader is not None
preflight = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = preflight
SPEC.loader.exec_module(preflight)

_ENV_KEYS = (
    *preflight.REQUIRED_PRESENT_ENV_KEYS,
    "LOCAL_WORKSPACE_SLACK_ASK_API_KEY",
    "LOCAL_WORKSPACE_SLACK_ASK_TIMEOUT_SECONDS",
)


@pytest.fixture(autouse=True)
def _clear_preflight_env() -> Generator[None, None, None]:
    saved = {key: os.environ[key] for key in _ENV_KEYS if key in os.environ}
    for key in _ENV_KEYS:
        os.environ.pop(key, None)
    yield
    for key in _ENV_KEYS:
        os.environ.pop(key, None)
    for key, value in saved.items():
        os.environ[key] = value


def _set_valid_env(*, api_key: str = "") -> None:
    os.environ["INTERGRAX_SLACK_CONVERSATION_ENABLED"] = "true"
    os.environ["INTERGRAX_SLACK_APP_TOKEN"] = "xapp-secret-token-value"
    os.environ["INTERGRAX_SLACK_BOT_TOKEN"] = "xoxb-secret-token-value"
    os.environ["LOCAL_WORKSPACE_SLACK_COMPANION_ENABLED"] = "true"
    os.environ["LOCAL_WORKSPACE_SLACK_APPROVED_TEAM_ID"] = "T0123456789"
    os.environ["LOCAL_WORKSPACE_SLACK_APPROVED_USER_ID"] = "U9876543210"
    os.environ["LOCAL_WORKSPACE_SLACK_TENANT_ID"] = "tenant-demo"
    os.environ["LOCAL_WORKSPACE_SLACK_ACTIVE_WORKSPACE_ID"] = "ws-demo"
    os.environ["LOCAL_WORKSPACE_SLACK_ASK_BASE_URL"] = "http://127.0.0.1:8020/"
    os.environ["LOCAL_WORKSPACE_SLACK_ASK_API_KEY"] = api_key
    os.environ["LOCAL_WORKSPACE_SLACK_ASK_TIMEOUT_SECONDS"] = "30"


def _json_response(status_code: int, payload: dict[str, Any]) -> httpx.Response:
    return httpx.Response(
        status_code,
        headers={"content-type": "application/json"},
        content=json.dumps(payload).encode("utf-8"),
    )


def _ok_workspace_payload() -> dict[str, Any]:
    return {
        "workspaces": [
            {
                "workspace_id": "ws-demo",
                "tenant_id": "tenant-demo",
                "name": "Demo",
                "description": "",
                "status": "active",
                "created_at": "2026-01-01T00:00:00Z",
                "updated_at": "2026-01-01T00:00:00Z",
            }
        ]
    }


def _ok_sources_payload(*, ready: bool = True) -> dict[str, Any]:
    source: dict[str, Any] = {
        "source_id": "src-1",
        "workspace_id": "ws-demo",
        "source_type": "local_folder",
        "path": "/secret/local/path/docs",
        "status": "ready" if ready else "registered",
        "recursive": True,
        "created_at": "2026-01-01T00:00:00Z",
        "last_sync_at": "2026-01-01T01:00:00Z" if ready else None,
    }
    return {"sources": [source]}


def _ask_payload(status: str) -> dict[str, Any]:
    return {
        "run_id": "ask-run-1",
        "workspace_id": "ws-demo",
        "status": status,
        "question": "SECRET QUESTION MUST NOT APPEAR",
        "answer": "SECRET ANSWER MUST NOT APPEAR",
        "citations": [
            {
                "evidence_id": "e1",
                "document_id": "d1",
                "source_id": "src-1",
                "workspace_id": "ws-demo",
                "source_path": "/secret/local/path/docs/proof.txt",
                "file_name": "proof.txt",
                "excerpt": "SECRET EXCERPT MUST NOT APPEAR",
            }
        ],
        "created_at": "2026-01-01T00:00:00Z",
        "completed_at": "2026-01-01T00:00:01Z",
        "error": None,
    }


def _transport(
    *,
    readiness_ok: bool = True,
    workspaces: dict[str, Any] | None = None,
    workspaces_status: int = 200,
    sources: dict[str, Any] | None = None,
    ask: dict[str, Any] | None = None,
    ask_status: int = 200,
    captured_headers: list[dict[str, str]] | None = None,
) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        if captured_headers is not None:
            captured_headers.append(dict(request.headers))
        path = request.url.path
        if path.endswith("/v1/local_workspace/readiness"):
            if not readiness_ok:
                return httpx.Response(503, json={"ready": False, "accepts_new_work": False})
            return httpx.Response(200, json={"ready": True, "accepts_new_work": True})
        if path.endswith("/v1/local_workspace/workspaces") and request.method == "GET":
            return _json_response(workspaces_status, workspaces or {"workspaces": []})
        if path.endswith("/sources") and request.method == "GET":
            return _json_response(200, sources or {"sources": []})
        if path.endswith("/ask") and request.method == "POST":
            return _json_response(ask_status, ask or _ask_payload("completed"))
        return httpx.Response(404, json={"detail": "missing"})

    return httpx.MockTransport(handler)


def _emit_text(result: Any) -> str:
    return "\n".join([*result.lines, f"PRECHECK={result.status}", f"reason={result.reason}"])


def test_missing_env_blocked() -> None:
    result = preflight.run_preflight(load_dotenv=False)
    assert result.status == "BLOCKED"
    assert result.reason == "missing_env"
    assert result.emit() == preflight.EXIT_BLOCKED
    text = _emit_text(result)
    assert "missing_env=" in text
    assert "LOCAL_WORKSPACE_SLACK_COMPANION_ENABLED" in text


def test_companion_disabled_blocked() -> None:
    _set_valid_env()
    os.environ["LOCAL_WORKSPACE_SLACK_COMPANION_ENABLED"] = "false"
    result = preflight.run_preflight(load_dotenv=False)
    assert result.status == "BLOCKED"
    assert result.reason == "companion_disabled"


def test_platform_transport_disabled_blocked() -> None:
    _set_valid_env()
    os.environ["INTERGRAX_SLACK_CONVERSATION_ENABLED"] = "false"
    result = preflight.run_preflight(load_dotenv=False)
    assert result.status == "BLOCKED"
    assert result.reason == "platform_transport_disabled"


def test_invalid_team_id_blocked() -> None:
    _set_valid_env()
    os.environ["LOCAL_WORKSPACE_SLACK_APPROVED_TEAM_ID"] = "W0123456789"
    result = preflight.run_preflight(load_dotenv=False)
    assert result.status == "BLOCKED"
    assert result.reason == "invalid_team_id_format"


def test_invalid_user_id_blocked() -> None:
    _set_valid_env()
    os.environ["LOCAL_WORKSPACE_SLACK_APPROVED_USER_ID"] = "B0123456789"
    result = preflight.run_preflight(load_dotenv=False)
    assert result.status == "BLOCKED"
    assert result.reason == "invalid_user_id_format"


def test_invalid_url_blocked() -> None:
    _set_valid_env()
    os.environ["LOCAL_WORKSPACE_SLACK_ASK_BASE_URL"] = "ftp://127.0.0.1:8020/"
    result = preflight.run_preflight(load_dotenv=False)
    assert result.status == "BLOCKED"
    assert result.reason == "invalid_ask_base_url"


def test_host_unreachable_blocked() -> None:
    _set_valid_env()

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    result = preflight.run_preflight(
        load_dotenv=False,
        transport=httpx.MockTransport(handler),
    )
    assert result.status == "BLOCKED"
    assert result.reason == "host_unreachable"
    assert "host_reachable=false" in result.lines


def test_workspace_list_http_error_blocked() -> None:
    _set_valid_env()
    result = preflight.run_preflight(
        load_dotenv=False,
        transport=_transport(workspaces_status=500, workspaces={"detail": "error"}),
    )
    assert result.status == "BLOCKED"
    assert result.reason == "workspace_list_http_500"


def test_workspace_missing_blocked() -> None:
    _set_valid_env()
    result = preflight.run_preflight(
        load_dotenv=False,
        transport=_transport(workspaces={"workspaces": []}, sources=_ok_sources_payload()),
    )
    assert result.status == "BLOCKED"
    assert result.reason == "workspace_missing"


def test_workspace_without_evidence_blocked() -> None:
    _set_valid_env()
    result = preflight.run_preflight(
        load_dotenv=False,
        transport=_transport(
            workspaces=_ok_workspace_payload(),
            sources={"sources": []},
        ),
    )
    assert result.status == "BLOCKED"
    assert result.reason == "workspace_without_evidence"


def test_partial_when_question_omitted() -> None:
    _set_valid_env()
    result = preflight.run_preflight(
        load_dotenv=False,
        transport=_transport(
            workspaces=_ok_workspace_payload(),
            sources=_ok_sources_payload(),
        ),
    )
    assert result.status == "PARTIAL"
    assert result.reason == "question_not_provided"
    assert result.emit() == preflight.EXIT_PARTIAL
    assert "ask_preflight=not_run" in result.lines


def test_pass_when_ask_completed() -> None:
    _set_valid_env(api_key="dev-api-key-secret")
    captured: list[dict[str, str]] = []
    result = preflight.run_preflight(
        question="What is the LKW live proof verification code?",
        load_dotenv=False,
        transport=_transport(
            workspaces=_ok_workspace_payload(),
            sources=_ok_sources_payload(),
            ask=_ask_payload("completed"),
            captured_headers=captured,
        ),
    )
    assert result.status == "PASS"
    assert result.emit() == preflight.EXIT_PASS
    assert "ask_status=completed" in result.lines
    assert any(h.get("x-api-key") == "dev-api-key-secret" for h in captured)


def test_insufficient_evidence_blocked() -> None:
    _set_valid_env()
    result = preflight.run_preflight(
        question="What is the LKW live proof verification code?",
        load_dotenv=False,
        transport=_transport(
            workspaces=_ok_workspace_payload(),
            sources=_ok_sources_payload(),
            ask=_ask_payload("insufficient_evidence"),
        ),
    )
    assert result.status == "BLOCKED"
    assert result.reason == "ask_status_insufficient_evidence"


def test_ask_failed_blocked() -> None:
    _set_valid_env()
    result = preflight.run_preflight(
        question="What is the LKW live proof verification code?",
        load_dotenv=False,
        transport=_transport(
            workspaces=_ok_workspace_payload(),
            sources=_ok_sources_payload(),
            ask=_ask_payload("failed"),
        ),
    )
    assert result.status == "BLOCKED"
    assert result.reason == "ask_status_failed"


def test_output_redacts_secrets_and_document_content() -> None:
    _set_valid_env(api_key="dev-api-key-secret")
    result = preflight.run_preflight(
        question="What is the LKW live proof verification code?",
        load_dotenv=False,
        transport=_transport(
            workspaces=_ok_workspace_payload(),
            sources=_ok_sources_payload(),
            ask=_ask_payload("completed"),
        ),
    )
    text = _emit_text(result)
    assert "xapp-secret-token-value" not in text
    assert "xoxb-secret-token-value" not in text
    assert "dev-api-key-secret" not in text
    assert "SECRET ANSWER" not in text
    assert "SECRET QUESTION" not in text
    assert "SECRET EXCERPT" not in text
    assert "/secret/local/path" not in text
    assert "proof.txt" in text
    assert "ask_api_key_configured=true" in text


def test_env_example_contains_product_companion_keys() -> None:
    content = ENV_EXAMPLE_PATH.read_text(encoding="utf-8")
    for key in SLACK_COMPANION_PRODUCT_ENV_KEYS:
        assert key in content, f"missing {key} in .env.example"
    assert "INTERGRAX_SLACK_CONVERSATION_ENABLED" in content
    assert "LKW Slack Ask companion" in content
    assert "LOCAL_WORKSPACE_SLACK_ASK_TIMEOUT_SECONDS" in content
    lowered = content.lower()
    assert "platform transport" in lowered or "socket mode transport" in lowered
    assert "lkw-slack-workflow-1b" in lowered
