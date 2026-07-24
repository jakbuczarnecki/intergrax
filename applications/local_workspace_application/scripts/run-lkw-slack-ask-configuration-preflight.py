# © Artur Czarnecki. All rights reserved.

"""Operator preflight for LKW-SLACK-WORKFLOW-1A configuration closure.

Validates env contract, host readiness, tenant/workspace evidence, and optional
Ask HTTP — without printing tokens, API keys, answers, excerpts, or local paths.

Usage:

  uv run python \\
    applications/local_workspace_application/scripts/run-lkw-slack-ask-configuration-preflight.py

  uv run python \\
    applications/local_workspace_application/scripts/run-lkw-slack-ask-configuration-preflight.py \\
    --question "What is the LKW live proof verification code?"

Exit codes:
  0 = PRECHECK=PASS
  1 = PRECHECK=PARTIAL
  2 = PRECHECK=BLOCKED
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse

import httpx

_APP_DIR = Path(__file__).resolve().parents[1]
_APPLICATIONS_ROOT = _APP_DIR.parent
_ENV_FILE = _APP_DIR / ".env"
if str(_APPLICATIONS_ROOT) not in sys.path:
    sys.path.insert(0, str(_APPLICATIONS_ROOT))

from local_workspace_application.slack_companion.companion import (  # noqa: E402
    SLACK_COMPANION_PRODUCT_ENV_KEYS,
)

PLATFORM_SLACK_ENV_KEYS: tuple[str, ...] = (
    "INTERGRAX_SLACK_CONVERSATION_ENABLED",
    "INTERGRAX_SLACK_APP_TOKEN",
    "INTERGRAX_SLACK_BOT_TOKEN",
)

REQUIRED_PRESENT_ENV_KEYS: tuple[str, ...] = (
    *PLATFORM_SLACK_ENV_KEYS,
    "LOCAL_WORKSPACE_SLACK_COMPANION_ENABLED",
    "LOCAL_WORKSPACE_SLACK_APPROVED_TEAM_ID",
    "LOCAL_WORKSPACE_SLACK_APPROVED_USER_ID",
    "LOCAL_WORKSPACE_SLACK_TENANT_ID",
    "LOCAL_WORKSPACE_SLACK_ACTIVE_WORKSPACE_ID",
    "LOCAL_WORKSPACE_SLACK_ASK_BASE_URL",
)

SECRET_ENV_KEYS: frozenset[str] = frozenset(
    {
        "INTERGRAX_SLACK_APP_TOKEN",
        "INTERGRAX_SLACK_BOT_TOKEN",
        "LOCAL_WORKSPACE_SLACK_ASK_API_KEY",
    }
)

EXIT_PASS = 0
EXIT_PARTIAL = 1
EXIT_BLOCKED = 2


@dataclass
class PreflightResult:
    status: str = "BLOCKED"
    reason: str = ""
    lines: list[str] = field(default_factory=list)

    def emit(self) -> int:
        for line in self.lines:
            print(line)
        print(f"PRECHECK={self.status}")
        if self.reason:
            print(f"reason={self.reason}")
        if self.status == "PASS":
            return EXIT_PASS
        if self.status == "PARTIAL":
            return EXIT_PARTIAL
        return EXIT_BLOCKED


def _load_dotenv_file(path: Path) -> None:
    """Load ``.env`` into ``os.environ`` without overriding existing values."""
    if not path.is_file():
        return
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(path, override=False)


def _env(name: str) -> str:
    return (os.environ.get(name) or "").strip()


def _truthy(name: str) -> bool:
    return _env(name).lower() in {"1", "true", "yes", "on"}


def _mask_id(value: str, *, prefix_hint: str = "") -> str:
    text = value.strip()
    if not text:
        return ""
    if len(text) <= 4:
        return f"{prefix_hint or text[:1]}…"
    return f"{text[0]}…{text[-3:]}"


def _is_http_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _headers(*, tenant_id: str, api_key: str) -> dict[str, str]:
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "X-Tenant-Id": tenant_id,
    }
    if api_key:
        headers["X-API-Key"] = api_key
    return headers


def _join(base_url: str, path: str) -> str:
    return urljoin(base_url.rstrip("/") + "/", path.lstrip("/"))


def validate_env_contract() -> tuple[dict[str, str] | None, PreflightResult]:
    """Return ``(config, result)``. ``config`` is None when BLOCKED on env/format."""
    result = PreflightResult()
    missing = [name for name in REQUIRED_PRESENT_ENV_KEYS if not _env(name)]
    if missing:
        result.lines.append("env_complete=false")
        result.lines.append("missing_env=" + ",".join(missing))
        result.status = "BLOCKED"
        result.reason = "missing_env"
        return None, result

    if not _truthy("LOCAL_WORKSPACE_SLACK_COMPANION_ENABLED"):
        result.lines.append("env_complete=true")
        result.lines.append("companion_enabled=false")
        result.status = "BLOCKED"
        result.reason = "companion_disabled"
        return None, result

    if not _truthy("INTERGRAX_SLACK_CONVERSATION_ENABLED"):
        result.lines.append("env_complete=true")
        result.lines.append("platform_transport_enabled=false")
        result.status = "BLOCKED"
        result.reason = "platform_transport_disabled"
        return None, result

    team_id = _env("LOCAL_WORKSPACE_SLACK_APPROVED_TEAM_ID")
    user_id = _env("LOCAL_WORKSPACE_SLACK_APPROVED_USER_ID")
    tenant_id = _env("LOCAL_WORKSPACE_SLACK_TENANT_ID")
    workspace_id = _env("LOCAL_WORKSPACE_SLACK_ACTIVE_WORKSPACE_ID")
    base_url = _env("LOCAL_WORKSPACE_SLACK_ASK_BASE_URL")
    api_key = _env("LOCAL_WORKSPACE_SLACK_ASK_API_KEY")
    timeout_raw = _env("LOCAL_WORKSPACE_SLACK_ASK_TIMEOUT_SECONDS") or "60"

    if not team_id.startswith("T"):
        result.lines.append("format_ok=false")
        result.status = "BLOCKED"
        result.reason = "invalid_team_id_format"
        return None, result
    if not user_id.startswith("U"):
        result.lines.append("format_ok=false")
        result.status = "BLOCKED"
        result.reason = "invalid_user_id_format"
        return None, result
    if not tenant_id:
        result.lines.append("format_ok=false")
        result.status = "BLOCKED"
        result.reason = "empty_tenant_id"
        return None, result
    if not workspace_id:
        result.lines.append("format_ok=false")
        result.status = "BLOCKED"
        result.reason = "empty_workspace_id"
        return None, result
    if not _is_http_url(base_url):
        result.lines.append("format_ok=false")
        result.status = "BLOCKED"
        result.reason = "invalid_ask_base_url"
        return None, result
    try:
        timeout = float(timeout_raw)
    except ValueError:
        result.lines.append("format_ok=false")
        result.status = "BLOCKED"
        result.reason = "invalid_timeout"
        return None, result
    if timeout <= 0:
        result.lines.append("format_ok=false")
        result.status = "BLOCKED"
        result.reason = "invalid_timeout"
        return None, result

    result.lines.extend(
        [
            "env_complete=true",
            "companion_enabled=true",
            "platform_transport_enabled=true",
            "format_ok=true",
            f"approved_team_id={_mask_id(team_id)}",
            f"approved_user_id={_mask_id(user_id)}",
            f"tenant_id={tenant_id}",
            f"workspace_id={workspace_id}",
            f"ask_base_url_scheme={urlparse(base_url).scheme}",
            f"ask_timeout_seconds={timeout}",
            f"ask_api_key_configured={'true' if api_key else 'false'}",
        ]
    )
    config = {
        "team_id": team_id,
        "user_id": user_id,
        "tenant_id": tenant_id,
        "workspace_id": workspace_id,
        "base_url": base_url,
        "api_key": api_key,
        "timeout": str(timeout),
    }
    return config, result


def check_host(
    config: dict[str, str],
    result: PreflightResult,
    *,
    transport: httpx.BaseTransport | None = None,
) -> bool:
    base_url = config["base_url"]
    timeout = float(config["timeout"])
    readiness_url = _join(base_url, "v1/local_workspace/readiness")
    try:
        with httpx.Client(timeout=min(timeout, 30.0), transport=transport) as client:
            response = client.get(readiness_url)
            body: dict[str, Any]
            try:
                payload = response.json()
                body = payload if isinstance(payload, dict) else {}
            except Exception:  # noqa: BLE001
                body = {}
    except httpx.HTTPError:
        result.lines.append("host_reachable=false")
        result.status = "BLOCKED"
        result.reason = "host_unreachable"
        return False

    reachable = response.status_code == 200
    ready = bool(body.get("ready") is True)
    accepts = bool(body.get("accepts_new_work") is True)
    healthy = reachable and ready and accepts
    result.lines.append(f"host_reachable={'true' if reachable else 'false'}")
    result.lines.append(
        f"core_readiness={'healthy' if healthy else 'unhealthy'}"
    )
    if not healthy:
        result.status = "BLOCKED"
        result.reason = "readiness_unhealthy" if reachable else "host_unreachable"
        return False
    return True


def _source_has_evidence(source: dict[str, Any]) -> bool:
    status = str(source.get("status") or "").strip().lower()
    if status == "ready":
        return True
    if source.get("last_sync_at"):
        return True
    return False


def check_workspace(
    config: dict[str, str],
    result: PreflightResult,
    *,
    transport: httpx.BaseTransport | None = None,
) -> bool:
    base_url = config["base_url"]
    tenant_id = config["tenant_id"]
    workspace_id = config["workspace_id"]
    api_key = config["api_key"]
    timeout = float(config["timeout"])
    headers = _headers(tenant_id=tenant_id, api_key=api_key)
    list_url = _join(base_url, "v1/local_workspace/workspaces")

    try:
        with httpx.Client(timeout=min(timeout, 30.0), transport=transport) as client:
            listed = client.get(list_url, headers=headers)
            if listed.status_code < 200 or listed.status_code >= 300:
                result.lines.append("workspace_list_ok=false")
                result.status = "BLOCKED"
                result.reason = f"workspace_list_http_{listed.status_code}"
                return False
            try:
                list_body = listed.json()
            except Exception:  # noqa: BLE001
                result.lines.append("workspace_list_ok=false")
                result.status = "BLOCKED"
                result.reason = "workspace_list_parse_error"
                return False
            workspaces = list_body.get("workspaces") if isinstance(list_body, dict) else None
            if not isinstance(workspaces, list):
                result.lines.append("workspace_list_ok=false")
                result.status = "BLOCKED"
                result.reason = "workspace_list_parse_error"
                return False

            match: dict[str, Any] | None = None
            for item in workspaces:
                if not isinstance(item, dict):
                    continue
                if str(item.get("workspace_id") or "") == workspace_id:
                    match = item
                    break
            if match is None:
                result.lines.append("workspace_found=false")
                result.status = "BLOCKED"
                result.reason = "workspace_missing"
                return False

            match_tenant = str(match.get("tenant_id") or "")
            if match_tenant and match_tenant != tenant_id:
                result.lines.append("workspace_found=true")
                result.lines.append("workspace_tenant_match=false")
                result.status = "BLOCKED"
                result.reason = "workspace_tenant_mismatch"
                return False

            status = str(match.get("status") or "").strip().lower() or "unknown"
            if status != "active":
                result.lines.append("workspace_found=true")
                result.lines.append(f"workspace_status={status}")
                result.status = "BLOCKED"
                result.reason = "workspace_not_askable"
                return False

            sources_url = _join(
                base_url,
                f"v1/local_workspace/workspaces/{workspace_id}/sources",
            )
            sources_resp = client.get(sources_url, headers=headers)
            if sources_resp.status_code < 200 or sources_resp.status_code >= 300:
                result.lines.append("workspace_found=true")
                result.lines.append(f"workspace_status={status}")
                result.status = "BLOCKED"
                result.reason = f"sources_http_{sources_resp.status_code}"
                return False
            try:
                sources_body = sources_resp.json()
            except Exception:  # noqa: BLE001
                result.lines.append("workspace_found=true")
                result.lines.append(f"workspace_status={status}")
                result.status = "BLOCKED"
                result.reason = "sources_parse_error"
                return False
            sources = sources_body.get("sources") if isinstance(sources_body, dict) else None
            if not isinstance(sources, list):
                result.lines.append("workspace_found=true")
                result.lines.append(f"workspace_status={status}")
                result.status = "BLOCKED"
                result.reason = "sources_parse_error"
                return False
            source_count = len(sources)
            evidence_count = sum(
                1 for src in sources if isinstance(src, dict) and _source_has_evidence(src)
            )
            result.lines.append("workspace_found=true")
            result.lines.append(f"workspace_status={status}")
            result.lines.append(f"source_count={source_count}")
            result.lines.append(f"evidence_source_count={evidence_count}")
            if source_count < 1 or evidence_count < 1:
                result.status = "BLOCKED"
                result.reason = "workspace_without_evidence"
                return False
            return True
    except httpx.HTTPError:
        result.lines.append("workspace_list_ok=false")
        result.status = "BLOCKED"
        result.reason = "workspace_list_transport_error"
        return False


def run_ask_preflight(
    config: dict[str, str],
    *,
    question: str,
    result: PreflightResult,
    transport: httpx.BaseTransport | None = None,
) -> bool:
    base_url = config["base_url"]
    tenant_id = config["tenant_id"]
    workspace_id = config["workspace_id"]
    api_key = config["api_key"]
    timeout = float(config["timeout"])
    url = _join(base_url, f"v1/local_workspace/workspaces/{workspace_id}/ask")
    headers = _headers(tenant_id=tenant_id, api_key=api_key)
    body = {"question": question, "limit": 10}

    try:
        with httpx.Client(timeout=timeout, transport=transport) as client:
            response = client.post(url, headers=headers, json=body)
    except httpx.HTTPError:
        result.lines.append("ask_preflight=error")
        result.status = "BLOCKED"
        result.reason = "ask_transport_error"
        return False

    if response.status_code < 200 or response.status_code >= 300:
        result.lines.append("ask_preflight=error")
        result.status = "BLOCKED"
        result.reason = f"ask_http_{response.status_code}"
        return False

    try:
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError("not_object")
    except Exception:  # noqa: BLE001
        result.lines.append("ask_preflight=error")
        result.status = "BLOCKED"
        result.reason = "ask_parse_error"
        return False

    status = str(payload.get("status") or "").strip()
    run_id = str(payload.get("run_id") or "").strip()
    citations = payload.get("citations")
    citation_count = len(citations) if isinstance(citations, list) else 0
    safe_names: list[str] = []
    if isinstance(citations, list):
        for item in citations:
            if not isinstance(item, dict):
                continue
            name = str(item.get("file_name") or "").strip()
            if name and name not in safe_names:
                safe_names.append(name)

    result.lines.append("ask_preflight=run")
    result.lines.append(f"ask_run_id={run_id or 'missing'}")
    result.lines.append(f"ask_status={status or 'missing'}")
    result.lines.append(f"citation_count={citation_count}")
    if safe_names:
        # file_name only — never source_path / excerpt / answer / question
        result.lines.append("safe_file_names=" + ",".join(safe_names[:20]))

    if status == "completed":
        return True
    if status in {"insufficient_evidence", "failed"}:
        result.status = "BLOCKED"
        result.reason = f"ask_status_{status}"
        return False
    result.status = "BLOCKED"
    result.reason = "ask_status_unexpected"
    return False


def run_preflight(
    *,
    question: str | None = None,
    transport: httpx.BaseTransport | None = None,
    load_dotenv: bool = True,
) -> PreflightResult:
    if load_dotenv:
        _load_dotenv_file(_ENV_FILE)

    config, result = validate_env_contract()
    if config is None:
        return result

    if not check_host(config, result, transport=transport):
        return result
    if not check_workspace(config, result, transport=transport):
        return result

    if not question or not question.strip():
        result.lines.append("ask_preflight=not_run")
        result.lines.append("reason=question_not_provided")
        result.status = "PARTIAL"
        result.reason = "question_not_provided"
        return result

    if not run_ask_preflight(
        config, question=question.strip(), result=result, transport=transport
    ):
        return result

    result.status = "PASS"
    result.reason = ""
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="LKW Slack Ask companion configuration preflight (no secrets)."
    )
    parser.add_argument(
        "--question",
        default=None,
        help="Optional Ask HTTP preflight question. Omit for PARTIAL config-only check.",
    )
    args = parser.parse_args(argv)
    return run_preflight(question=args.question).emit()


# Re-export for tests / .env.example contract.
PRODUCT_ENV_KEYS = SLACK_COMPANION_PRODUCT_ENV_KEYS
__all__ = [
    "PRODUCT_ENV_KEYS",
    "REQUIRED_PRESENT_ENV_KEYS",
    "SECRET_ENV_KEYS",
    "EXIT_PASS",
    "EXIT_PARTIAL",
    "EXIT_BLOCKED",
    "run_preflight",
    "validate_env_contract",
    "main",
]


if __name__ == "__main__":
    sys.exit(main())
