# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Unit tests for Microsoft Graph knowledge-read common foundation."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from intergrax.integrations.contracts.base import (
    IntegrationConfigurationError,
    IntegrationDependencyError,
)
from intergrax.integrations.providers.collaboration_suite.ms365_graph.client import GraphRestClient
from intergrax.integrations.providers.collaboration_suite.ms365_graph.config import (
    DEFAULT_GRAPH_BASE_URL,
    Ms365GraphIntegrationConfig,
)
from intergrax.integrations.providers.collaboration_suite.ms365_graph.knowledge_read import (
    MsGraphKnowledgeCollectionPage,
    MsGraphKnowledgeContinuation,
    MsGraphKnowledgeContinuationKind,
    MsGraphKnowledgeSyncResetRequired,
    MsGraphKnowledgeTransport,
    parse_msgraph_collection_page,
    validate_msgraph_continuation_url,
)

pytestmark = pytest.mark.unit

_GRAPH_BASE = DEFAULT_GRAPH_BASE_URL
_NEXT_LINK = (
    "https://graph.microsoft.com/v1.0/users/user-1/messages?"
    "$skiptoken=opaque-page-token-value"
)
_DELTA_LINK = (
    "https://graph.microsoft.com/v1.0/users/user-1/messages/delta?"
    "$deltatoken=opaque-delta-token-value"
)
_SECRET_TOKEN = "super-secret-skiptoken-value"


def _config() -> Ms365GraphIntegrationConfig:
    return Ms365GraphIntegrationConfig(
        tenant_id="tenant-123",
        client_id="client-456",
        client_secret="secret",
        graph_base_url=_GRAPH_BASE,
    )


def _page_payload(
    *,
    items: list[dict[str, Any]] | None = None,
    next_link: str | None = None,
    delta_link: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"value": items or [{"id": "item-1"}]}
    if next_link is not None:
        payload["@odata.nextLink"] = next_link
    if delta_link is not None:
        payload["@odata.deltaLink"] = delta_link
    return payload


def _mock_http(*, status_code: int = 200, json_payload: object | None = None) -> MagicMock:
    client = MagicMock()
    response = MagicMock()
    response.status_code = status_code
    response.json.return_value = json_payload if json_payload is not None else {}
    response.raise_for_status = MagicMock()
    client.get.return_value = response
    return client


# --- continuation URL validation ---


def test_validate_next_link_accepts_graph_path() -> None:
    validated = validate_msgraph_continuation_url(_NEXT_LINK, graph_base_url=_GRAPH_BASE)
    assert validated == _NEXT_LINK


def test_validate_delta_link_accepts_graph_path() -> None:
    validated = validate_msgraph_continuation_url(_DELTA_LINK, graph_base_url=_GRAPH_BASE)
    assert validated == _DELTA_LINK


def test_validate_preserves_opaque_query_string() -> None:
    url = (
        "https://graph.microsoft.com/v1.0/drives/drive-1/root/delta?"
        "$deltatoken=abc%3D%3D&other=1"
    )
    validated = validate_msgraph_continuation_url(url, graph_base_url=_GRAPH_BASE)
    assert validated == url
    assert "$deltatoken=abc%3D%3D&other=1" in validated


def test_validate_rejects_http_scheme() -> None:
    url = "http://graph.microsoft.com/v1.0/users"
    with pytest.raises(ValueError, match="invalid Microsoft Graph continuation URL") as exc:
        validate_msgraph_continuation_url(url, graph_base_url=_GRAPH_BASE)
    assert url not in str(exc.value)
    assert _SECRET_TOKEN not in str(exc.value)


def test_validate_rejects_foreign_host() -> None:
    url = "https://evil.example/v1.0/users"
    with pytest.raises(ValueError, match="invalid Microsoft Graph continuation URL") as exc:
        validate_msgraph_continuation_url(url, graph_base_url=_GRAPH_BASE)
    assert url not in str(exc.value)


def test_validate_rejects_foreign_port() -> None:
    url = "https://graph.microsoft.com:8443/v1.0/users"
    with pytest.raises(ValueError, match="invalid Microsoft Graph continuation URL"):
        validate_msgraph_continuation_url(url, graph_base_url=_GRAPH_BASE)


def test_validate_rejects_beta_path_for_v1_config() -> None:
    url = "https://graph.microsoft.com/beta/users"
    with pytest.raises(ValueError, match="invalid Microsoft Graph continuation URL"):
        validate_msgraph_continuation_url(url, graph_base_url=_GRAPH_BASE)


def test_validate_rejects_credentials_in_url() -> None:
    url = "https://user:password@graph.microsoft.com/v1.0/users"
    with pytest.raises(ValueError, match="invalid Microsoft Graph continuation URL") as exc:
        validate_msgraph_continuation_url(url, graph_base_url=_GRAPH_BASE)
    assert "password" not in str(exc.value)
    assert url not in str(exc.value)


def test_validate_rejects_fragment() -> None:
    url = "https://graph.microsoft.com/v1.0/users#fragment"
    with pytest.raises(ValueError, match="invalid Microsoft Graph continuation URL"):
        validate_msgraph_continuation_url(url, graph_base_url=_GRAPH_BASE)


def test_validate_rejects_relative_url() -> None:
    with pytest.raises(ValueError, match="invalid Microsoft Graph continuation URL"):
        validate_msgraph_continuation_url("/v1.0/users", graph_base_url=_GRAPH_BASE)


def test_validate_rejects_non_string() -> None:
    with pytest.raises(ValueError, match="invalid Microsoft Graph continuation URL"):
        validate_msgraph_continuation_url(123, graph_base_url=_GRAPH_BASE)


def test_continuation_model_hides_url_in_repr() -> None:
    continuation = MsGraphKnowledgeContinuation(
        kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
        url=_NEXT_LINK,
    )
    rendered = repr(continuation)
    assert _SECRET_TOKEN not in rendered
    assert "nextLink" not in rendered
    assert "skiptoken" not in rendered


# --- collection page parser ---


def test_parse_regular_last_page_without_link() -> None:
    page = parse_msgraph_collection_page(
        _page_payload(items=[{"id": "a"}]),
        graph_base_url=_GRAPH_BASE,
        delta_mode=False,
    )
    assert isinstance(page, MsGraphKnowledgeCollectionPage)
    assert page.items == ({"id": "a"},)
    assert page.continuation is None


def test_parse_regular_page_with_next_link() -> None:
    page = parse_msgraph_collection_page(
        _page_payload(next_link=_NEXT_LINK),
        graph_base_url=_GRAPH_BASE,
        delta_mode=False,
    )
    assert page.continuation is not None
    assert page.continuation.kind == MsGraphKnowledgeContinuationKind.NEXT_PAGE
    assert page.continuation.url == _NEXT_LINK


def test_parse_regular_page_rejects_delta_link() -> None:
    with pytest.raises(ValueError, match="unexpected Microsoft Graph knowledge response"):
        parse_msgraph_collection_page(
            _page_payload(delta_link=_DELTA_LINK),
            graph_base_url=_GRAPH_BASE,
            delta_mode=False,
        )


def test_parse_delta_page_with_next_link() -> None:
    page = parse_msgraph_collection_page(
        _page_payload(next_link=_NEXT_LINK),
        graph_base_url=_GRAPH_BASE,
        delta_mode=True,
    )
    assert page.continuation is not None
    assert page.continuation.kind == MsGraphKnowledgeContinuationKind.NEXT_PAGE


def test_parse_delta_page_with_delta_link() -> None:
    page = parse_msgraph_collection_page(
        _page_payload(delta_link=_DELTA_LINK),
        graph_base_url=_GRAPH_BASE,
        delta_mode=True,
    )
    assert page.continuation is not None
    assert page.continuation.kind == MsGraphKnowledgeContinuationKind.DELTA


def test_parse_delta_page_requires_link() -> None:
    with pytest.raises(ValueError, match="unexpected Microsoft Graph knowledge response"):
        parse_msgraph_collection_page(
            _page_payload(),
            graph_base_url=_GRAPH_BASE,
            delta_mode=True,
        )


def test_parse_rejects_both_links() -> None:
    with pytest.raises(ValueError, match="unexpected Microsoft Graph knowledge response") as exc:
        parse_msgraph_collection_page(
            _page_payload(next_link=_NEXT_LINK, delta_link=_DELTA_LINK),
            graph_base_url=_GRAPH_BASE,
            delta_mode=True,
        )
    assert _SECRET_TOKEN not in str(exc.value)


def test_parse_rejects_non_list_value() -> None:
    with pytest.raises(ValueError, match="unexpected Microsoft Graph knowledge response"):
        parse_msgraph_collection_page(
            {"value": "not-a-list"},
            graph_base_url=_GRAPH_BASE,
            delta_mode=False,
        )


def test_parse_rejects_non_dict_item() -> None:
    with pytest.raises(ValueError, match="unexpected Microsoft Graph knowledge response"):
        parse_msgraph_collection_page(
            {"value": ["bad"]},
            graph_base_url=_GRAPH_BASE,
            delta_mode=False,
        )


def test_parse_rejects_malformed_link() -> None:
    with pytest.raises(ValueError, match="invalid Microsoft Graph continuation URL") as exc:
        parse_msgraph_collection_page(
            _page_payload(next_link="https://evil.example/v1.0/users"),
            graph_base_url=_GRAPH_BASE,
            delta_mode=False,
        )
    assert "evil.example" not in str(exc.value)


def test_parse_error_does_not_leak_provider_token() -> None:
    token = "provider-token-should-not-appear"
    with pytest.raises(ValueError) as exc:
        parse_msgraph_collection_page(
            _page_payload(
                next_link=f"https://evil.example/v1.0/users?token={token}",
            ),
            graph_base_url=_GRAPH_BASE,
            delta_mode=False,
        )
    assert token not in str(exc.value)


# --- transport ---


def test_initial_get_uses_relative_path() -> None:
    http = _mock_http(json_payload={"value": []})
    transport = MsGraphKnowledgeTransport(_config(), http_client=http)
    payload = transport.get_initial_json(path="/users/user-1/messages", params={"$top": 10})
    assert payload == {"value": []}
    http.get.assert_called_once()
    assert http.get.call_args.args[0] == "/users/user-1/messages"
    assert http.get.call_args.kwargs["params"] == {"$top": 10}


def test_continuation_get_uses_full_url() -> None:
    http = _mock_http(json_payload={"value": []})
    transport = MsGraphKnowledgeTransport(_config(), http_client=http)
    continuation = MsGraphKnowledgeContinuation(
        kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
        url=_NEXT_LINK,
    )
    transport.get_continuation_json(continuation=continuation)
    http.get.assert_called_once()
    assert http.get.call_args.args[0] == _NEXT_LINK
    assert "params" not in http.get.call_args.kwargs or http.get.call_args.kwargs.get("params") is None


def test_graph_rest_client_wires_shared_http_client() -> None:
    http = _mock_http()
    client = GraphRestClient(_config(), http_client=http)
    assert client._knowledge_transport._http_client is http


def test_transport_exception_maps_to_dependency_error() -> None:
    http = MagicMock()
    http.get.side_effect = RuntimeError("network down")
    transport = MsGraphKnowledgeTransport(_config(), http_client=http)
    with pytest.raises(IntegrationDependencyError, match="dependency is unavailable") as exc:
        transport.get_initial_json(path="/users/user-1/messages")
    assert exc.value.__cause__ is None


def test_status_429_maps_to_dependency_error() -> None:
    http = _mock_http(status_code=429)
    transport = MsGraphKnowledgeTransport(_config(), http_client=http)
    with pytest.raises(IntegrationDependencyError, match="dependency failure"):
        transport.get_initial_json(path="/users/user-1/messages")


def test_status_503_maps_to_dependency_error() -> None:
    http = _mock_http(status_code=503)
    transport = MsGraphKnowledgeTransport(_config(), http_client=http)
    with pytest.raises(IntegrationDependencyError, match="dependency failure"):
        transport.get_initial_json(path="/users/user-1/messages")


@pytest.mark.parametrize("status_code", [400, 401, 403])
def test_status_4xx_configuration_maps(status_code: int) -> None:
    http = _mock_http(status_code=status_code)
    transport = MsGraphKnowledgeTransport(_config(), http_client=http)
    with pytest.raises(IntegrationConfigurationError, match="configuration failure"):
        transport.get_initial_json(path="/users/user-1/messages")


def test_status_404_configuration_by_default() -> None:
    http = _mock_http(status_code=404)
    transport = MsGraphKnowledgeTransport(_config(), http_client=http)
    with pytest.raises(IntegrationConfigurationError, match="configuration failure"):
        transport.get_initial_json(path="/users/user-1/messages")


def test_status_404_dependency_when_flag_set() -> None:
    http = _mock_http(status_code=404)
    transport = MsGraphKnowledgeTransport(_config(), http_client=http)
    with pytest.raises(IntegrationDependencyError, match="dependency failure"):
        transport.get_initial_json(
            path="/users/user-1/messages",
            not_found_is_dependency=True,
        )


def test_status_410_maps_to_sync_reset_required() -> None:
    http = _mock_http(status_code=410)
    transport = MsGraphKnowledgeTransport(_config(), http_client=http)
    with pytest.raises(MsGraphKnowledgeSyncResetRequired, match="must restart"):
        transport.get_initial_json(path="/users/user-1/messages/delta")


def test_json_decode_failure_is_safe() -> None:
    http = MagicMock()
    response = MagicMock()
    response.status_code = 200
    response.json.side_effect = ValueError("bad json")
    http.get.return_value = response
    transport = MsGraphKnowledgeTransport(_config(), http_client=http)
    with pytest.raises(ValueError, match="unexpected Microsoft Graph knowledge response") as exc:
        transport.get_initial_json(path="/users/user-1/messages")
    assert exc.value.__cause__ is None


def test_non_dict_json_payload_rejected() -> None:
    http = _mock_http(json_payload=["not", "a", "dict"])
    transport = MsGraphKnowledgeTransport(_config(), http_client=http)
    with pytest.raises(ValueError, match="unexpected Microsoft Graph knowledge response"):
        transport.get_initial_json(path="/users/user-1/messages")


def test_error_messages_do_not_leak_response_body_or_credentials() -> None:
    secret_body = "client-secret-body"
    http = MagicMock()
    response = MagicMock()
    response.status_code = 400
    response.json.return_value = {"error": {"message": secret_body}}
    http.get.return_value = response
    transport = MsGraphKnowledgeTransport(_config(), http_client=http)
    with pytest.raises(IntegrationConfigurationError) as exc:
        transport.get_initial_json(path="/users/user-1/messages")
    assert secret_body not in str(exc.value)
    assert _config().client_secret not in str(exc.value)


def test_transport_does_not_call_raise_for_status() -> None:
    http = _mock_http(json_payload={"value": []})
    transport = MsGraphKnowledgeTransport(_config(), http_client=http)
    transport.get_initial_json(path="/users/user-1/messages")
    http.get.return_value.raise_for_status.assert_not_called()
