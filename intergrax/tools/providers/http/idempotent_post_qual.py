# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Qualification-only idempotent HTTP POST — enforces provider idempotency key contract."""

from __future__ import annotations

from intergrax.integrations.contracts.http_client import HttpClientBackend
from intergrax.tools.providers.http.contracts import HttpRequestInput, HttpRequestOutput
from intergrax.tools.core.handler import WiringContextToolHandler
from intergrax.tools.execution_models import ToolExecutionRequest
from intergrax.tools.invocation_wiring_adapter import effective_wiring_for_request
from intergrax.tools.providers.provider_resource_enforcement import (
    ProviderResourceLimitExceeded,
    bound_http_response_body,
)
from intergrax.tools.registry.wiring import ToolWiringContext

HTTP_IDEMPOTENT_POST_QUAL_TOOL_ID = "http.idempotent_post.qual"

_IDEMPOTENCY_HEADER = "Idempotency-Key"
_IDEMPOTENCY_HEADER_LOWER = _IDEMPOTENCY_HEADER.lower()


def _normalize_idempotency_headers(
    headers: dict[str, str],
    *,
    authoritative_key: str,
) -> dict[str, str]:
    key = authoritative_key.strip()
    if not key:
        raise ValueError("idempotent_post_qual_requires_authoritative_idempotency_key")
    normalized: dict[str, str] = {}
    for header_name, header_value in headers.items():
        if header_name.lower() == _IDEMPOTENCY_HEADER_LOWER:
            if header_value.strip() and header_value.strip() != key:
                raise ValueError("idempotent_post_qual_idempotency_key_header_conflict")
            continue
        normalized[header_name] = header_value
    normalized[_IDEMPOTENCY_HEADER] = key
    return normalized


def _require_http_client(ctx: ToolWiringContext) -> HttpClientBackend:
    client = ctx.http_client or ctx.extras.get("http_client")
    if client is None:
        raise RuntimeError("http_client_not_configured")
    return client


def http_idempotent_post_qual(
    ctx: ToolWiringContext,
    params: HttpRequestInput,
    *,
    authoritative_idempotency_key: str | None = None,
) -> HttpRequestOutput:
    method = params.method.strip().upper()
    if method != "POST":
        raise ValueError("idempotent_post_qual_requires_post")
    if authoritative_idempotency_key is None:
        raise ValueError("idempotent_post_qual_requires_authoritative_idempotency_key")
    headers = _normalize_idempotency_headers(
        dict(params.headers),
        authoritative_key=authoritative_idempotency_key,
    )
    client = _require_http_client(ctx)
    try:
        response = client.request(
            method,
            params.url.strip(),
            headers=headers,
            body=params.body,
            timeout_s=params.timeout_s,
            max_response_bytes=params.max_response_bytes,
        )
    except Exception as exc:  # noqa: BLE001 — tool boundary
        raise RuntimeError(f"http_transport_error:{exc.__class__.__name__}") from exc
    truncated = response.body_truncated_at_transport
    body = response.body
    if not response.transport_byte_ceiling_enforced:
        try:
            body, output_truncated = bound_http_response_body(
                body,
                max_response_bytes=params.max_response_bytes,
            )
        except ProviderResourceLimitExceeded as exc:
            return HttpRequestOutput(success=False, error=exc.code)
        truncated = truncated or output_truncated
    return HttpRequestOutput(
        success=200 <= response.status_code < 400,
        status_code=response.status_code,
        body=body,
        headers=dict(response.headers),
        truncated=truncated,
    )


class HttpIdempotentPostQualHandler(
    WiringContextToolHandler[HttpRequestInput, HttpRequestOutput],
):
    def execute(self, request: ToolExecutionRequest[HttpRequestInput]) -> HttpRequestOutput:
        ctx = effective_wiring_for_request(request, self._ctx)
        return http_idempotent_post_qual(
            ctx,
            request.input,
            authoritative_idempotency_key=request.idempotency_key,
        )


def register_http_idempotent_post_qual_tool(registry: object, ctx: ToolWiringContext) -> None:
    """Register the sanctioned Class B qualification catalog tool (not generic http.request)."""
    from intergrax.tools.core.contracts import ToolContract, ToolRiskLevel
    from intergrax.tools.providers.http.class_b_qualification_provenance import (
        build_sanctioned_http_idempotent_post_qual_activation,
    )
    from intergrax.tools.registry.runtime import ToolRegistry

    if not isinstance(registry, ToolRegistry):
        raise TypeError("registry must be ToolRegistry")
    activation = build_sanctioned_http_idempotent_post_qual_activation(
        logical_tool_id=HTTP_IDEMPOTENT_POST_QUAL_TOOL_ID,
    )
    registry.register(
        ToolContract(
            tool_id=HTTP_IDEMPOTENT_POST_QUAL_TOOL_ID,
            name=HTTP_IDEMPOTENT_POST_QUAL_TOOL_ID,
            description=(
                "Execute one allowlisted idempotent HTTP POST with mandatory "
                "Idempotency-Key header (R4 qualification path)."
            ),
            description_short="Idempotent HTTP POST (qual).",
            input_schema=HttpRequestInput,
            output_schema=HttpRequestOutput,
            error_mapping={},
            side_effects=True,
            category="http",
            risk_level=ToolRiskLevel.HIGH,
            tags=("http", "integration", "network", "tool-life-qual"),
        ),
        HttpIdempotentPostQualHandler(ctx),
        activation=activation,
    )
