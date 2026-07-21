#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""Shared cross-platform LKW interaction client.

OS wrappers are thin launchers only. Payload construction, HTTP, and
normalized result emission live here.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Mapping

SCHEMA_VERSION = "local_workspace.os_interaction_adapter_result.v1"
INTAKE_ENDPOINT = "/v1/interactions/intake"
CLIENT_RUNTIME = "python"

EXIT_OK = 0
EXIT_INVALID_INPUT = 2
EXIT_REQUEST_FAILED = 3

ERROR_INVALID_ADAPTER_INPUT = "invalid_adapter_input"
ERROR_UNSUPPORTED_OS_FAMILY = "unsupported_os_family"
ERROR_OS_CONTRACT_MISMATCH = "os_contract_mismatch"
ERROR_RUNTIME_OS_MISMATCH = "runtime_os_mismatch"
ERROR_INTERACTION_REQUEST_FAILED = "interaction_request_failed"
ERROR_INTERACTION_RESPONSE_INVALID = "interaction_response_invalid"


@dataclass(frozen=True, slots=True)
class OsInteractionContract:
    os_family: str
    adapter_id: str
    source: str
    wrapper_runtime: str


OS_CONTRACTS: Mapping[str, OsInteractionContract] = {
    "windows": OsInteractionContract(
        os_family="windows",
        adapter_id="lkw.windows_powershell",
        source="windows_powershell",
        wrapper_runtime="windows_powershell",
    ),
    "linux": OsInteractionContract(
        os_family="linux",
        adapter_id="lkw.linux_shell",
        source="linux_shell",
        wrapper_runtime="posix_sh",
    ),
    "macos": OsInteractionContract(
        os_family="macos",
        adapter_id="lkw.macos_shell",
        source="macos_shell",
        wrapper_runtime="posix_sh",
    ),
}


class ClientError(Exception):
    def __init__(
        self, error_id: str, *, exit_code: int, http_status: int | None = None
    ):
        super().__init__(error_id)
        self.error_id = error_id
        self.exit_code = exit_code
        self.http_status = http_status


def detect_runtime_os_family(system_name: str | None = None) -> str:
    """Normalize ``platform.system()`` to a supported OS family token."""
    name = (system_name if system_name is not None else platform.system()).strip()
    mapping = {
        "Windows": "windows",
        "Linux": "linux",
        "Darwin": "macos",
    }
    detected = mapping.get(name)
    if detected is None:
        raise ClientError(ERROR_UNSUPPORTED_OS_FAMILY, exit_code=EXIT_INVALID_INPUT)
    return detected


def validate_os_contract(
    *,
    os_family: str,
    adapter_id: str,
    source: str,
    wrapper_runtime: str,
) -> OsInteractionContract:
    """Fail closed unless the declared identity matches a frozen OS contract."""
    family = os_family.strip()
    if not family:
        raise ClientError(ERROR_INVALID_ADAPTER_INPUT, exit_code=EXIT_INVALID_INPUT)
    contract = OS_CONTRACTS.get(family)
    if contract is None:
        raise ClientError(ERROR_UNSUPPORTED_OS_FAMILY, exit_code=EXIT_INVALID_INPUT)

    adapter = adapter_id.strip()
    src = source.strip()
    wrapper = wrapper_runtime.strip()
    if not adapter or not src or not wrapper:
        raise ClientError(ERROR_INVALID_ADAPTER_INPUT, exit_code=EXIT_INVALID_INPUT)

    if (
        adapter != contract.adapter_id
        or src != contract.source
        or wrapper != contract.wrapper_runtime
    ):
        raise ClientError(ERROR_OS_CONTRACT_MISMATCH, exit_code=EXIT_INVALID_INPUT)
    return contract


def validate_runtime_os_matches_declared(
    declared_os_family: str,
    *,
    runtime_os_family: str | None = None,
) -> str:
    """Require declared OS family to match the actual Python runtime OS."""
    actual = (
        runtime_os_family
        if runtime_os_family is not None
        else detect_runtime_os_family()
    )
    if actual != declared_os_family.strip():
        raise ClientError(ERROR_RUNTIME_OS_MISMATCH, exit_code=EXIT_INVALID_INPUT)
    return actual


def resolve_base_url(raw_base_url: str) -> str:
    resolved = raw_base_url.strip()
    if not resolved:
        resolved = os.environ.get("LOCAL_WORKSPACE_BACKEND_BASE_URL", "").strip()
    if not resolved:
        resolved = "http://127.0.0.1:8020"
    while resolved.endswith("/"):
        resolved = resolved[:-1]
    return resolved


def parse_metadata_json(raw: str) -> dict[str, Any]:
    text = raw if raw.strip() else "{}"
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ClientError(
            ERROR_INVALID_ADAPTER_INPUT, exit_code=EXIT_INVALID_INPUT
        ) from exc
    if not isinstance(parsed, dict):
        raise ClientError(ERROR_INVALID_ADAPTER_INPUT, exit_code=EXIT_INVALID_INPUT)
    return parsed


def build_intake_payload(
    *,
    contract: OsInteractionContract,
    message: str,
    tenant_id: str,
    user_id: str,
    capability: str,
    session_id: str,
    interaction_id: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "tenant_id": tenant_id.strip(),
        "user_id": user_id.strip(),
        "message": message,
        "source": contract.source,
        "metadata": metadata,
    }
    if capability.strip():
        body["capability"] = capability.strip()
    if session_id.strip():
        body["session_id"] = session_id.strip()
    if interaction_id.strip():
        body["interaction_id"] = interaction_id.strip()
    return body


def build_intake_url(*, base_url: str, tenant_id: str) -> str:
    encoded_tenant = urllib.parse.quote(tenant_id.strip(), safe="")
    query = f"execute=true&tenant={encoded_tenant}"
    return f"{base_url}{INTAKE_ENDPOINT}?{query}"


def post_interaction_intake(
    *,
    url: str,
    payload: dict[str, Any],
    timeout_seconds: float,
) -> Any:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json; charset=utf-8",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            body = response.read().decode("utf-8")
            status = int(response.status)
    except urllib.error.HTTPError as exc:
        raise ClientError(
            ERROR_INTERACTION_REQUEST_FAILED,
            exit_code=EXIT_REQUEST_FAILED,
            http_status=int(exc.code),
        ) from exc
    except urllib.error.URLError as exc:
        raise ClientError(
            ERROR_INTERACTION_REQUEST_FAILED,
            exit_code=EXIT_REQUEST_FAILED,
        ) from exc
    except TimeoutError as exc:
        raise ClientError(
            ERROR_INTERACTION_REQUEST_FAILED,
            exit_code=EXIT_REQUEST_FAILED,
        ) from exc

    if not (200 <= status < 300):
        raise ClientError(
            ERROR_INTERACTION_REQUEST_FAILED,
            exit_code=EXIT_REQUEST_FAILED,
            http_status=status,
        )
    if not body.strip():
        raise ClientError(
            ERROR_INTERACTION_RESPONSE_INVALID,
            exit_code=EXIT_REQUEST_FAILED,
        )
    try:
        return json.loads(body)
    except json.JSONDecodeError as exc:
        raise ClientError(
            ERROR_INTERACTION_RESPONSE_INVALID,
            exit_code=EXIT_REQUEST_FAILED,
        ) from exc


def build_success_result(
    *,
    contract: OsInteractionContract,
    response: Any,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "result": "PASS",
        "os_family": contract.os_family,
        "os_version": platform.version(),
        "architecture": platform.machine() or platform.architecture()[0],
        "client_runtime": CLIENT_RUNTIME,
        "wrapper_runtime": contract.wrapper_runtime,
        "adapter_id": contract.adapter_id,
        "source": contract.source,
        "endpoint": INTAKE_ENDPOINT,
        "execute": True,
        "response": response,
    }


def emit_failure(error: ClientError) -> int:
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "result": "FAIL",
        "error_id": error.error_id,
    }
    if error.http_status is not None:
        payload["http_status"] = error.http_status
    sys.stderr.write(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"
    )
    return error.exit_code


def emit_success(result: dict[str, Any]) -> int:
    sys.stdout.write(
        json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n"
    )
    return EXIT_OK


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Shared LKW OS interaction intake client.",
    )
    parser.add_argument("--os-family", required=True)
    parser.add_argument("--adapter-id", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--wrapper-runtime", required=True)
    parser.add_argument("--message", required=True)
    parser.add_argument("--base-url", default="")
    parser.add_argument("--capability", default="")
    parser.add_argument("--tenant-id", default="default")
    parser.add_argument("--user-id", default="os-user")
    parser.add_argument("--session-id", default="")
    parser.add_argument("--interaction-id", default="")
    parser.add_argument("--metadata-json", default="{}")
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    return parser.parse_args(argv)


def run_client(argv: list[str] | None = None) -> int:
    try:
        args = _parse_args(argv)
        message = args.message
        if not isinstance(message, str) or not message.strip():
            raise ClientError(ERROR_INVALID_ADAPTER_INPUT, exit_code=EXIT_INVALID_INPUT)
        if not str(args.tenant_id).strip() or not str(args.user_id).strip():
            raise ClientError(ERROR_INVALID_ADAPTER_INPUT, exit_code=EXIT_INVALID_INPUT)
        if float(args.timeout_seconds) <= 0:
            raise ClientError(ERROR_INVALID_ADAPTER_INPUT, exit_code=EXIT_INVALID_INPUT)

        contract = validate_os_contract(
            os_family=args.os_family,
            adapter_id=args.adapter_id,
            source=args.source,
            wrapper_runtime=args.wrapper_runtime,
        )
        validate_runtime_os_matches_declared(contract.os_family)
        metadata = parse_metadata_json(str(args.metadata_json))
        base_url = resolve_base_url(str(args.base_url))
        payload = build_intake_payload(
            contract=contract,
            message=message,
            tenant_id=str(args.tenant_id),
            user_id=str(args.user_id),
            capability=str(args.capability),
            session_id=str(args.session_id),
            interaction_id=str(args.interaction_id),
            metadata=metadata,
        )
        url = build_intake_url(base_url=base_url, tenant_id=str(args.tenant_id))
        response = post_interaction_intake(
            url=url,
            payload=payload,
            timeout_seconds=float(args.timeout_seconds),
        )
        return emit_success(build_success_result(contract=contract, response=response))
    except ClientError as exc:
        return emit_failure(exc)
    except (OSError, ValueError, TypeError):
        return emit_failure(
            ClientError(ERROR_INVALID_ADAPTER_INPUT, exit_code=EXIT_INVALID_INPUT)
        )


def main() -> int:
    return run_client()


if __name__ == "__main__":
    sys.exit(main())
