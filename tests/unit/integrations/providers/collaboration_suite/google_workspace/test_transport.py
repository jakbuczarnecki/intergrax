# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import pytest

from intergrax.integrations.providers.collaboration_suite.google_workspace.contracts import (
    GoogleWorkspaceBinaryPayload,
    GoogleWorkspaceSourceKind,
)
from intergrax.integrations.providers.collaboration_suite.google_workspace.transport import (
    GoogleWorkspaceApiError,
    GoogleWorkspaceCollectionPage,
    GoogleWorkspaceErrorKind,
    GoogleWorkspaceHttpTransport,
    GoogleWorkspacePageToken,
    GoogleWorkspaceRetryPolicy,
    parse_google_workspace_collection_page,
)

_APPROVED_ROOTS = {
    GoogleWorkspaceSourceKind.DRIVE: "https://www.googleapis.com/drive/v3",
    GoogleWorkspaceSourceKind.DOCS: "https://docs.googleapis.com/v1",
    GoogleWorkspaceSourceKind.SHEETS: "https://sheets.googleapis.com/v4",
    GoogleWorkspaceSourceKind.SLIDES: "https://slides.googleapis.com/v1",
    GoogleWorkspaceSourceKind.CALENDAR: "https://www.googleapis.com/calendar/v3",
    GoogleWorkspaceSourceKind.MAIL: "https://gmail.googleapis.com/gmail/v1",
    GoogleWorkspaceSourceKind.CHAT: "https://chat.googleapis.com/v1",
}


@dataclass
class _FakeResponse:
    status_code: int
    headers: dict[str, str] = field(default_factory=dict)
    content: bytes = b"{}"

    def json(self) -> object:
        return json.loads(self.content)


@dataclass
class _RecordingExecutor:
    calls: list[dict[str, object]] = field(default_factory=list)
    responses: list[_FakeResponse] = field(default_factory=list)
    errors: list[Exception] = field(default_factory=list)

    def get(
        self,
        *,
        url: str,
        params: Mapping[str, object] | None,
        headers: Mapping[str, str],
        timeout_seconds: float,
    ) -> _FakeResponse:
        self.calls.append(
            {
                "url": url,
                "params": None if params is None else dict(params),
                "headers": dict(headers),
                "timeout_seconds": timeout_seconds,
            }
        )
        if self.errors:
            raise self.errors.pop(0)
        return self.responses.pop(0)


def _transport(
    executor: _RecordingExecutor,
    *,
    policy: GoogleWorkspaceRetryPolicy | None = None,
    sleeper: list[float] | None = None,
    jitter_values: list[float] | None = None,
) -> GoogleWorkspaceHttpTransport:
    sleeps: list[float] = sleeper if sleeper is not None else []

    def _sleeper(seconds: float) -> None:
        sleeps.append(seconds)

    jitter_queue = list(jitter_values or [0.0])

    def _jitter() -> float:
        if jitter_queue:
            return jitter_queue.pop(0)
        return 0.0

    return GoogleWorkspaceHttpTransport(
        executor=executor,
        retry_policy=policy or GoogleWorkspaceRetryPolicy(),
        sleeper=_sleeper,
        jitter_source=_jitter,
    )


def test_all_source_kinds_map_to_approved_roots() -> None:
    executor = _RecordingExecutor()
    transport = _transport(executor)
    for kind, root in _APPROVED_ROOTS.items():
        executor.responses.append(_FakeResponse(200, content=b'{"ok": true}'))
        transport.get_json(source_kind=kind, relative_path="/files")
        assert executor.calls[-1]["url"] == f"{root}/files"


@pytest.mark.parametrize(
    ("relative_path", "forbidden"),
    [
        ("https://evil.example/files", "absolute"),
        ("//evil.example/files", "protocol_relative"),
        ("/a/../b", "traversal"),
        ("/files?x=1", "query_in_path"),
    ],
)
def test_unsafe_relative_paths_rejected(relative_path: str, forbidden: str) -> None:
    executor = _RecordingExecutor()
    transport = _transport(executor)
    with pytest.raises(GoogleWorkspaceApiError) as exc_info:
        transport.get_json(
            source_kind=GoogleWorkspaceSourceKind.DRIVE,
            relative_path=relative_path,
        )
    assert exc_info.value.kind is GoogleWorkspaceErrorKind.INVALID_REQUEST
    assert executor.calls == []


@pytest.mark.parametrize(
    "param_name",
    ["access_token", "OAUTH_TOKEN", "Authorization", "API_KEY", "refresh_token"],
)
def test_forbidden_query_parameters_rejected(param_name: str) -> None:
    executor = _RecordingExecutor()
    transport = _transport(executor)
    with pytest.raises(GoogleWorkspaceApiError) as exc_info:
        transport.get_json(
            source_kind=GoogleWorkspaceSourceKind.DRIVE,
            relative_path="/files",
            params={param_name: "secret"},
        )
    assert exc_info.value.safe_reason == "forbidden_query_parameter"
    assert executor.calls == []


@pytest.mark.parametrize("header_name", ["Authorization", "cookie", "X-Goog-Api-Key"])
def test_forbidden_headers_rejected(header_name: str) -> None:
    executor = _RecordingExecutor()
    transport = _transport(executor)
    with pytest.raises(GoogleWorkspaceApiError) as exc_info:
        transport.get_json(
            source_kind=GoogleWorkspaceSourceKind.DRIVE,
            relative_path="/files",
            headers={header_name: "secret"},
        )
    assert exc_info.value.safe_reason == "forbidden_header"
    assert executor.calls == []


def test_successful_get_json_uses_executor_once_and_applies_accept() -> None:
    executor = _RecordingExecutor(responses=[_FakeResponse(200, content=b'{"value": 1}')])
    transport = _transport(executor)
    params = {"pageSize": 10}
    headers = {"X-Custom": "safe"}
    result = transport.get_json(
        source_kind=GoogleWorkspaceSourceKind.DRIVE,
        relative_path="/files",
        params=params,
        headers=headers,
    )
    assert result == {"value": 1}
    assert len(executor.calls) == 1
    call = executor.calls[0]
    assert call["url"] == "https://www.googleapis.com/drive/v3/files"
    assert call["params"] == {"pageSize": 10}
    assert call["headers"]["Accept"] == "application/json"
    assert call["headers"]["X-Custom"] == "safe"
    assert call["timeout_seconds"] == 30.0
    assert params == {"pageSize": 10}
    assert headers == {"X-Custom": "safe"}


def test_invalid_status_type_rejected() -> None:
    class _BadStatus:
        status_code = "200"
        headers: dict[str, str] = {}
        content = b"{}"

        def json(self) -> object:
            return {}

    executor = _RecordingExecutor()
    executor.responses.append(_BadStatus())  # type: ignore[arg-type]
    transport = _transport(executor)
    with pytest.raises(GoogleWorkspaceApiError) as exc_info:
        transport.get_json(source_kind=GoogleWorkspaceSourceKind.DRIVE, relative_path="/files")
    assert exc_info.value.kind is GoogleWorkspaceErrorKind.MALFORMED_RESPONSE


def test_invalid_status_range_rejected() -> None:
    executor = _RecordingExecutor(responses=[_FakeResponse(99, content=b"{}")])
    transport = _transport(executor)
    with pytest.raises(GoogleWorkspaceApiError) as exc_info:
        transport.get_json(source_kind=GoogleWorkspaceSourceKind.DRIVE, relative_path="/files")
    assert exc_info.value.kind is GoogleWorkspaceErrorKind.MALFORMED_RESPONSE


def test_non_bytes_content_rejected() -> None:
    class _BadContent:
        status_code = 200
        headers: dict[str, str] = {}
        content = "{}"

        def json(self) -> object:
            return {}

    executor = _RecordingExecutor()
    executor.responses.append(_BadContent())  # type: ignore[arg-type]
    transport = _transport(executor)
    with pytest.raises(GoogleWorkspaceApiError) as exc_info:
        transport.get_json(source_kind=GoogleWorkspaceSourceKind.DRIVE, relative_path="/files")
    assert exc_info.value.kind is GoogleWorkspaceErrorKind.MALFORMED_RESPONSE


def test_oversized_response_rejected_before_json_decode() -> None:
    policy = GoogleWorkspaceRetryPolicy(max_response_bytes=4)
    executor = _RecordingExecutor(responses=[_FakeResponse(200, content=b"12345")])
    transport = _transport(executor, policy=policy)
    with pytest.raises(GoogleWorkspaceApiError) as exc_info:
        transport.get_json(source_kind=GoogleWorkspaceSourceKind.DRIVE, relative_path="/files")
    assert exc_info.value.safe_reason == "response_too_large"


def test_top_level_non_object_json_rejected() -> None:
    executor = _RecordingExecutor(responses=[_FakeResponse(200, content=b"[]")])
    transport = _transport(executor)
    with pytest.raises(GoogleWorkspaceApiError) as exc_info:
        transport.get_json(source_kind=GoogleWorkspaceSourceKind.DRIVE, relative_path="/files")
    assert exc_info.value.error_code == "GOOGLE_WORKSPACE_MALFORMED_RESPONSE"


@pytest.mark.parametrize(
    ("status_code", "body", "expected_kind", "retryable"),
    [
        (400, b'{"error": {"message": "bad"}}', GoogleWorkspaceErrorKind.INVALID_REQUEST, False),
        (401, b"{}", GoogleWorkspaceErrorKind.AUTHENTICATION, False),
        (403, b'{"error": {"errors": [{"reason": "forbidden"}]}}', GoogleWorkspaceErrorKind.AUTHORIZATION, False),
        (
            403,
            b'{"error": {"errors": [{"reason": "rateLimitExceeded"}]}}',
            GoogleWorkspaceErrorKind.RATE_LIMITED,
            True,
        ),
        (404, b"{}", GoogleWorkspaceErrorKind.NOT_FOUND, False),
        (408, b"{}", GoogleWorkspaceErrorKind.TEMPORARY, True),
        (429, b"{}", GoogleWorkspaceErrorKind.RATE_LIMITED, True),
        (500, b"{}", GoogleWorkspaceErrorKind.TEMPORARY, True),
        (502, b"{}", GoogleWorkspaceErrorKind.TEMPORARY, True),
        (503, b"{}", GoogleWorkspaceErrorKind.TEMPORARY, True),
        (504, b"{}", GoogleWorkspaceErrorKind.TEMPORARY, True),
        (302, b"{}", GoogleWorkspaceErrorKind.UNEXPECTED_REDIRECT, False),
    ],
)
def test_error_mapping(
    status_code: int,
    body: bytes,
    expected_kind: GoogleWorkspaceErrorKind,
    retryable: bool,
) -> None:
    executor = _RecordingExecutor(responses=[_FakeResponse(status_code, content=body)])
    policy = GoogleWorkspaceRetryPolicy(max_attempts=1)
    transport = _transport(executor, policy=policy)
    with pytest.raises(GoogleWorkspaceApiError) as exc_info:
        transport.get_json(source_kind=GoogleWorkspaceSourceKind.DRIVE, relative_path="/files")
    error = exc_info.value
    assert error.kind is expected_kind
    assert error.retryable is retryable
    assert "bad" not in str(error)
    assert error.attempts == 1


def test_executor_exception_maps_to_temporary_without_message() -> None:
    executor = _RecordingExecutor(errors=[RuntimeError("network secret failure")])
    transport = _transport(executor)
    with pytest.raises(GoogleWorkspaceApiError) as exc_info:
        transport.get_json(source_kind=GoogleWorkspaceSourceKind.DRIVE, relative_path="/files")
    assert exc_info.value.kind is GoogleWorkspaceErrorKind.TEMPORARY
    assert "network secret failure" not in str(exc_info.value)


def test_non_retryable_errors_do_not_retry() -> None:
    executor = _RecordingExecutor(responses=[_FakeResponse(400, content=b"{}")])
    sleeps: list[float] = []
    transport = _transport(executor, sleeper=sleeps)
    with pytest.raises(GoogleWorkspaceApiError):
        transport.get_json(source_kind=GoogleWorkspaceSourceKind.DRIVE, relative_path="/files")
    assert len(executor.calls) == 1
    assert sleeps == []


def test_retryable_errors_retry_until_max_attempts_without_final_sleep() -> None:
    policy = GoogleWorkspaceRetryPolicy(max_attempts=3, base_backoff_seconds=1.0, max_backoff_seconds=8.0)
    executor = _RecordingExecutor(
        responses=[
            _FakeResponse(503, content=b"{}"),
            _FakeResponse(503, content=b"{}"),
            _FakeResponse(503, content=b"{}"),
        ]
    )
    sleeps: list[float] = []
    transport = _transport(executor, policy=policy, sleeper=sleeps, jitter_values=[0.5, 0.5])
    with pytest.raises(GoogleWorkspaceApiError) as exc_info:
        transport.get_json(source_kind=GoogleWorkspaceSourceKind.DRIVE, relative_path="/files")
    assert len(executor.calls) == 3
    assert sleeps == [0.5, 1.0]
    assert exc_info.value.attempts == 3


def test_successful_second_attempt_returns_payload() -> None:
    policy = GoogleWorkspaceRetryPolicy(max_attempts=3)
    executor = _RecordingExecutor(
        responses=[
            _FakeResponse(429, content=b"{}", headers={"Retry-After": "2"}),
            _FakeResponse(200, content=b'{"ok": true}'),
        ]
    )
    sleeps: list[float] = []
    transport = _transport(executor, policy=policy, sleeper=sleeps)
    result = transport.get_json(source_kind=GoogleWorkspaceSourceKind.DRIVE, relative_path="/files")
    assert result == {"ok": True}
    assert len(executor.calls) == 2
    assert sleeps == [2.0]


def test_retry_after_is_bounded() -> None:
    policy = GoogleWorkspaceRetryPolicy(max_attempts=2, max_retry_after_seconds=5)
    executor = _RecordingExecutor(
        responses=[
            _FakeResponse(429, content=b"{}", headers={"Retry-After": "120"}),
            _FakeResponse(200, content=b'{"ok": true}'),
        ]
    )
    sleeps: list[float] = []
    transport = _transport(executor, policy=policy, sleeper=sleeps)
    transport.get_json(source_kind=GoogleWorkspaceSourceKind.DRIVE, relative_path="/files")
    assert sleeps == [5.0]


def test_invalid_retry_after_is_ignored() -> None:
    policy = GoogleWorkspaceRetryPolicy(max_attempts=2, base_backoff_seconds=2.0)
    executor = _RecordingExecutor(
        responses=[
            _FakeResponse(429, content=b"{}", headers={"Retry-After": "-1"}),
            _FakeResponse(200, content=b'{"ok": true}'),
        ]
    )
    sleeps: list[float] = []
    transport = _transport(executor, policy=policy, sleeper=sleeps, jitter_values=[0.0])
    transport.get_json(source_kind=GoogleWorkspaceSourceKind.DRIVE, relative_path="/files")
    assert sleeps == [0.0]


def test_retry_after_http_date_honored() -> None:
    future = datetime.now(timezone.utc) + timedelta(seconds=3)
    policy = GoogleWorkspaceRetryPolicy(max_attempts=2, max_retry_after_seconds=30.0)
    executor = _RecordingExecutor(
        responses=[
            _FakeResponse(
                429,
                content=b"{}",
                headers={"Retry-After": future.strftime("%a, %d %b %Y %H:%M:%S GMT")},
            ),
            _FakeResponse(200, content=b'{"ok": true}'),
        ]
    )
    sleeps: list[float] = []
    transport = _transport(executor, policy=policy, sleeper=sleeps)
    transport.get_json(source_kind=GoogleWorkspaceSourceKind.DRIVE, relative_path="/files")
    assert len(sleeps) == 1
    assert 0 < sleeps[0] <= 30.0


def test_security_fields_not_exposed_in_error() -> None:
    body = json.dumps(
        {
            "error": {
                "message": "provider secret message",
                "errors": [{"reason": "rateLimitExceeded"}],
            }
        }
    ).encode()
    executor = _RecordingExecutor(responses=[_FakeResponse(429, content=body)])
    transport = _transport(executor)
    with pytest.raises(GoogleWorkspaceApiError) as exc_info:
        transport.get_json(
            source_kind=GoogleWorkspaceSourceKind.DRIVE,
            relative_path="/files",
            params={"pageToken": "opaque-token-value"},
        )
    error = exc_info.value
    assert "provider secret message" not in str(error)
    assert "opaque-token-value" not in str(error)
    assert "access_token" not in str(error)


def test_page_token_hidden_from_repr() -> None:
    token = GoogleWorkspacePageToken(value="super-secret-token")
    assert "super-secret-token" not in repr(token)


def test_parse_collection_page_valid_without_token() -> None:
    payload = {"files": [{"id": "1"}, {"id": "2"}]}
    original = json.loads(json.dumps(payload))
    page = parse_google_workspace_collection_page(payload, items_field="files")
    assert isinstance(page, GoogleWorkspaceCollectionPage)
    assert page.items == ({"id": "1"}, {"id": "2"})
    assert page.next_page_token is None
    assert payload == original


def test_parse_collection_page_valid_with_token() -> None:
    page = parse_google_workspace_collection_page(
        {"files": [{"id": "1"}], "nextPageToken": "token-1"},
        items_field="files",
    )
    assert page.next_page_token is not None
    assert page.next_page_token.value == "token-1"


def test_parse_collection_page_preserves_item_order() -> None:
    page = parse_google_workspace_collection_page(
        {"files": [{"id": "a"}, {"id": "b"}, {"id": "c"}]},
        items_field="files",
    )
    assert [item["id"] for item in page.items] == ["a", "b", "c"]


@pytest.mark.parametrize(
    ("payload", "items_field"),
    [
        ("not-a-dict", "files"),
        ({"files": "nope"}, "files"),
        ({"other": []}, "files"),
        ({"files": ["bad"]}, "files"),
        ({"files": [{}], "nextPageToken": "   "}, "files"),
        ({"files": [{}], "nextPageToken": "x" * 5000}, "files"),
        ({"files": [{}], "nextPageToken": "bad\x00token"}, "files"),
    ],
)
def test_parse_collection_page_malformed_rejected(payload: object, items_field: str) -> None:
    with pytest.raises((ValueError, TypeError)):
        parse_google_workspace_collection_page(payload, items_field=items_field)


class _MissingStatusResponse:
    headers: dict[str, str] = {}
    content = b"{}"

    def json(self) -> object:
        return {}


class _RaisingStatusResponse:
    @property
    def status_code(self) -> int:
        raise RuntimeError("status secret failure")

    headers: dict[str, str] = {}
    content = b"{}"

    def json(self) -> object:
        return {}


class _MissingHeadersResponse:
    status_code = 200
    content = b"{}"

    def json(self) -> object:
        return {}


class _RaisingHeadersResponse:
    status_code = 200
    content = b"{}"

    @property
    def headers(self) -> dict[str, str]:
        raise RuntimeError("headers secret failure")

    def json(self) -> object:
        return {}


class _MissingContentResponse:
    status_code = 200
    headers: dict[str, str] = {}

    def json(self) -> object:
        return {}


class _RaisingContentResponse:
    status_code = 200
    headers: dict[str, str] = {}

    @property
    def content(self) -> bytes:
        raise RuntimeError("content secret failure")

    def json(self) -> object:
        return {}


@pytest.mark.parametrize(
    ("response", "secret_message"),
    [
        (_MissingStatusResponse(), None),
        (_RaisingStatusResponse(), "status secret failure"),
        (_MissingHeadersResponse(), None),
        (_RaisingHeadersResponse(), "headers secret failure"),
        (_MissingContentResponse(), None),
        (_RaisingContentResponse(), "content secret failure"),
    ],
)
def test_malformed_response_object_boundaries(
    response: object,
    secret_message: str | None,
) -> None:
    executor = _RecordingExecutor()
    executor.responses.append(response)  # type: ignore[arg-type]
    transport = _transport(executor)
    with pytest.raises(GoogleWorkspaceApiError) as exc_info:
        transport.get_json(source_kind=GoogleWorkspaceSourceKind.DRIVE, relative_path="/files")
    assert exc_info.value.kind is GoogleWorkspaceErrorKind.MALFORMED_RESPONSE
    if secret_message is not None:
        assert secret_message not in str(exc_info.value)


def test_successful_2xx_invalid_utf_becomes_malformed_response() -> None:
    executor = _RecordingExecutor(responses=[_FakeResponse(200, content=b"\xff\xfe")])
    transport = _transport(executor)
    with pytest.raises(GoogleWorkspaceApiError) as exc_info:
        transport.get_json(source_kind=GoogleWorkspaceSourceKind.DRIVE, relative_path="/files")
    assert exc_info.value.kind is GoogleWorkspaceErrorKind.MALFORMED_RESPONSE
    assert exc_info.value.safe_reason == "invalid_json"
    assert b"\xff" not in str(exc_info.value).encode()


def test_error_response_invalid_utf_classified_by_http_status() -> None:
    executor = _RecordingExecutor(responses=[_FakeResponse(401, content=b"\xff\xfe")])
    policy = GoogleWorkspaceRetryPolicy(max_attempts=1)
    transport = _transport(executor, policy=policy)
    with pytest.raises(GoogleWorkspaceApiError) as exc_info:
        transport.get_json(source_kind=GoogleWorkspaceSourceKind.DRIVE, relative_path="/files")
    assert exc_info.value.kind is GoogleWorkspaceErrorKind.AUTHENTICATION
    assert b"\xff" not in str(exc_info.value).encode()
    assert repr(exc_info.value).find("\\xff") == -1


@pytest.mark.parametrize(
    "policy_kwargs",
    [
        {"max_attempts": True},
        {"max_response_bytes": True},
        {"request_timeout_seconds": True},
        {"request_timeout_seconds": float("nan")},
        {"base_backoff_seconds": float("inf")},
        {"max_backoff_seconds": float("-inf")},
    ],
)
def test_retry_policy_rejects_invalid_numeric_types(policy_kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        GoogleWorkspaceRetryPolicy(**policy_kwargs)  # type: ignore[arg-type]


def test_retry_policy_valid_boundary_values_preserved() -> None:
    policy = GoogleWorkspaceRetryPolicy(
        max_attempts=5,
        max_response_bytes=16_777_216,
        request_timeout_seconds=120,
        base_backoff_seconds=10,
        max_backoff_seconds=60,
        max_retry_after_seconds=120,
    )
    assert policy.max_attempts == 5
    assert policy.max_response_bytes == 16_777_216


@pytest.mark.parametrize(
    "invalid_jitter",
    [-0.1, 1.0, 2.0, float("nan"), float("inf"), True, "0.5", None],
)
def test_invalid_jitter_values_fail_without_sleep_or_retry(
    invalid_jitter: object,
) -> None:
    executor = _RecordingExecutor(responses=[_FakeResponse(503, content=b"{}")])
    policy = GoogleWorkspaceRetryPolicy(max_attempts=3)
    sleeps: list[float] = []

    def _bad_jitter() -> object:
        return invalid_jitter

    transport = GoogleWorkspaceHttpTransport(
        executor=executor,
        retry_policy=policy,
        sleeper=lambda seconds: sleeps.append(seconds),
        jitter_source=_bad_jitter,  # type: ignore[arg-type]
    )
    with pytest.raises(GoogleWorkspaceApiError) as exc_info:
        transport.get_json(source_kind=GoogleWorkspaceSourceKind.DRIVE, relative_path="/files")
    assert exc_info.value.safe_reason == "invalid_jitter_source"
    assert len(executor.calls) == 1
    assert sleeps == []


def test_jitter_exception_is_sanitized_without_sleep_or_retry() -> None:
    executor = _RecordingExecutor(responses=[_FakeResponse(503, content=b"{}")])
    policy = GoogleWorkspaceRetryPolicy(max_attempts=3)
    sleeps: list[float] = []

    def _raising_jitter() -> float:
        raise RuntimeError("secret jitter message")

    transport = GoogleWorkspaceHttpTransport(
        executor=executor,
        retry_policy=policy,
        sleeper=lambda seconds: sleeps.append(seconds),
        jitter_source=_raising_jitter,
    )
    with pytest.raises(GoogleWorkspaceApiError) as exc_info:
        transport.get_json(source_kind=GoogleWorkspaceSourceKind.DRIVE, relative_path="/files")
    assert exc_info.value.safe_reason == "invalid_jitter_source"
    assert "secret jitter message" not in str(exc_info.value)
    assert len(executor.calls) == 1
    assert sleeps == []


def test_invalid_source_kind_does_not_raise_key_error() -> None:
    executor = _RecordingExecutor()
    transport = _transport(executor)
    with pytest.raises(GoogleWorkspaceApiError) as exc_info:
        transport.get_json(source_kind="drive", relative_path="/files")  # type: ignore[arg-type]
    assert exc_info.value.kind is GoogleWorkspaceErrorKind.INVALID_REQUEST
    assert executor.calls == []


def test_non_string_query_parameter_key_fails_safely() -> None:
    executor = _RecordingExecutor()
    transport = _transport(executor)
    with pytest.raises(GoogleWorkspaceApiError) as exc_info:
        transport.get_json(
            source_kind=GoogleWorkspaceSourceKind.DRIVE,
            relative_path="/files",
            params={1: "value"},  # type: ignore[dict-item]
        )
    assert exc_info.value.safe_reason == "invalid_query_parameter"
    assert executor.calls == []


def test_non_string_header_name_fails_safely() -> None:
    executor = _RecordingExecutor()
    transport = _transport(executor)
    with pytest.raises(GoogleWorkspaceApiError) as exc_info:
        transport.get_json(
            source_kind=GoogleWorkspaceSourceKind.DRIVE,
            relative_path="/files",
            headers={1: "safe"},  # type: ignore[dict-item]
        )
    assert exc_info.value.safe_reason == "invalid_header"
    assert executor.calls == []


def test_non_string_header_value_fails_safely() -> None:
    executor = _RecordingExecutor()
    transport = _transport(executor)
    with pytest.raises(GoogleWorkspaceApiError) as exc_info:
        transport.get_json(
            source_kind=GoogleWorkspaceSourceKind.DRIVE,
            relative_path="/files",
            headers={"X-Custom": 123},  # type: ignore[dict-item]
        )
    assert exc_info.value.safe_reason == "invalid_header"
    assert executor.calls == []


_SECRET_MAPPING_MESSAGE = "secret mapping message"


class _RaisingItemsMapping(Mapping[object, object]):
    def __getitem__(self, key: object) -> object:
        return "value"

    def __iter__(self):
        return iter(())

    def __len__(self) -> int:
        return 1

    def items(self) -> object:
        raise RuntimeError(_SECRET_MAPPING_MESSAGE)


class _MalformedResponseHeaders:
    status_code = 200
    content = b"{}"

    @property
    def headers(self) -> _RaisingItemsMapping:
        return _RaisingItemsMapping()

    def json(self) -> object:
        return {}


def test_params_mapping_exception_fails_closed() -> None:
    executor = _RecordingExecutor()
    transport = _transport(executor)
    with pytest.raises(GoogleWorkspaceApiError) as exc_info:
        transport.get_json(
            source_kind=GoogleWorkspaceSourceKind.DRIVE,
            relative_path="/files",
            params=_RaisingItemsMapping(),
        )
    assert exc_info.value.error_code == "GOOGLE_WORKSPACE_INVALID_REQUEST"
    assert exc_info.value.safe_reason == "invalid_query_parameter"
    assert executor.calls == []
    assert _SECRET_MAPPING_MESSAGE not in str(exc_info.value)


def test_headers_mapping_exception_fails_closed() -> None:
    executor = _RecordingExecutor()
    transport = _transport(executor)
    with pytest.raises(GoogleWorkspaceApiError) as exc_info:
        transport.get_json(
            source_kind=GoogleWorkspaceSourceKind.DRIVE,
            relative_path="/files",
            headers=_RaisingItemsMapping(),
        )
    assert exc_info.value.error_code == "GOOGLE_WORKSPACE_INVALID_REQUEST"
    assert exc_info.value.safe_reason == "invalid_header"
    assert executor.calls == []
    assert _SECRET_MAPPING_MESSAGE not in str(exc_info.value)


def test_response_headers_mapping_exception_fails_closed() -> None:
    executor = _RecordingExecutor(responses=[_MalformedResponseHeaders()])  # type: ignore[list-item]
    transport = _transport(executor)
    with pytest.raises(GoogleWorkspaceApiError) as exc_info:
        transport.get_json(source_kind=GoogleWorkspaceSourceKind.DRIVE, relative_path="/files")
    assert exc_info.value.error_code == "GOOGLE_WORKSPACE_MALFORMED_RESPONSE"
    assert exc_info.value.status_code == 200
    assert exc_info.value.attempts == 1
    assert exc_info.value.safe_reason == "invalid_response_headers"
    assert _SECRET_MAPPING_MESSAGE not in str(exc_info.value)


def test_malformed_headers_on_retry_preserves_attempt_and_status_metadata() -> None:
    class _BadHeadersOnSecondAttempt:
        status_code = 503
        content = b"{}"
        headers: dict[str, str] = {}

        def json(self) -> object:
            return {}

    class _MalformedHeaders503:
        status_code = 503
        content = b"{}"

        @property
        def headers(self) -> _RaisingItemsMapping:
            return _RaisingItemsMapping()

        def json(self) -> object:
            return {}

    policy = GoogleWorkspaceRetryPolicy(max_attempts=3)
    executor = _RecordingExecutor(
        responses=[
            _BadHeadersOnSecondAttempt(),
            _MalformedHeaders503(),
        ]
    )
    sleeps: list[float] = []
    transport = _transport(executor, policy=policy, sleeper=sleeps, jitter_values=[0.5])
    with pytest.raises(GoogleWorkspaceApiError) as exc_info:
        transport.get_json(source_kind=GoogleWorkspaceSourceKind.DRIVE, relative_path="/files")
    assert exc_info.value.error_code == "GOOGLE_WORKSPACE_MALFORMED_RESPONSE"
    assert exc_info.value.status_code == 503
    assert exc_info.value.attempts == 2
    assert exc_info.value.safe_reason == "invalid_response_headers"
    assert len(executor.calls) == 2
    assert len(sleeps) == 1
    assert _SECRET_MAPPING_MESSAGE not in str(exc_info.value)


_SECRET_TYPED_ERROR_FRAGMENT = "access_token=super-secret-value"


def _foreign_typed_api_error() -> GoogleWorkspaceApiError:
    return GoogleWorkspaceApiError(
        kind=GoogleWorkspaceErrorKind.TEMPORARY,
        status_code=503,
        retry_after_seconds=30,
        safe_reason=_SECRET_TYPED_ERROR_FRAGMENT,
        attempts=99,
    )


class _TypedErrorRaisingItemsMapping(Mapping[object, object]):
    def __getitem__(self, key: object) -> object:
        return "value"

    def __iter__(self):
        return iter(())

    def __len__(self) -> int:
        return 1

    def items(self) -> object:
        raise _foreign_typed_api_error()


class _LazyTypedErrorIterator:
    def __iter__(self):
        return self

    def __next__(self) -> tuple[str, str]:
        raise _foreign_typed_api_error()


class _LazyTypedErrorItemsMapping(Mapping[object, object]):
    def __getitem__(self, key: object) -> object:
        return "value"

    def __iter__(self):
        return iter(())

    def __len__(self) -> int:
        return 1

    def items(self) -> object:
        return _LazyTypedErrorIterator()


@pytest.mark.parametrize(
    "mapping",
    [
        _TypedErrorRaisingItemsMapping(),
        _LazyTypedErrorItemsMapping(),
    ],
)
def test_params_typed_mapping_exception_is_canonicalized(mapping: object) -> None:
    executor = _RecordingExecutor()
    transport = _transport(executor)
    with pytest.raises(GoogleWorkspaceApiError) as exc_info:
        transport.get_json(
            source_kind=GoogleWorkspaceSourceKind.DRIVE,
            relative_path="/files",
            params=mapping,  # type: ignore[arg-type]
        )
    error = exc_info.value
    assert error.error_code == "GOOGLE_WORKSPACE_INVALID_REQUEST"
    assert error.kind is GoogleWorkspaceErrorKind.INVALID_REQUEST
    assert error.safe_reason == "invalid_query_parameter"
    assert error.status_code is None
    assert error.attempts == 0
    assert error.retry_after_seconds is None
    assert executor.calls == []
    assert _SECRET_TYPED_ERROR_FRAGMENT not in error.safe_reason
    assert _SECRET_TYPED_ERROR_FRAGMENT not in str(error)
    assert _SECRET_TYPED_ERROR_FRAGMENT not in repr(error)


@pytest.mark.parametrize(
    "mapping",
    [
        _TypedErrorRaisingItemsMapping(),
        _LazyTypedErrorItemsMapping(),
    ],
)
def test_headers_typed_mapping_exception_is_canonicalized(mapping: object) -> None:
    executor = _RecordingExecutor()
    transport = _transport(executor)
    with pytest.raises(GoogleWorkspaceApiError) as exc_info:
        transport.get_json(
            source_kind=GoogleWorkspaceSourceKind.DRIVE,
            relative_path="/files",
            headers=mapping,  # type: ignore[arg-type]
        )
    error = exc_info.value
    assert error.error_code == "GOOGLE_WORKSPACE_INVALID_REQUEST"
    assert error.kind is GoogleWorkspaceErrorKind.INVALID_REQUEST
    assert error.safe_reason == "invalid_header"
    assert error.status_code is None
    assert error.attempts == 0
    assert error.retry_after_seconds is None
    assert executor.calls == []
    assert _SECRET_TYPED_ERROR_FRAGMENT not in error.safe_reason
    assert _SECRET_TYPED_ERROR_FRAGMENT not in str(error)
    assert _SECRET_TYPED_ERROR_FRAGMENT not in repr(error)


class _TypedErrorResponseHeaders:
    status_code = 401
    content = b"{}"

    @property
    def headers(self) -> _TypedErrorRaisingItemsMapping:
        return _TypedErrorRaisingItemsMapping()

    def json(self) -> object:
        return {}


def test_response_headers_typed_mapping_exception_is_canonicalized() -> None:
    executor = _RecordingExecutor(responses=[_TypedErrorResponseHeaders()])  # type: ignore[list-item]
    transport = _transport(executor)
    with pytest.raises(GoogleWorkspaceApiError) as exc_info:
        transport.get_json(source_kind=GoogleWorkspaceSourceKind.DRIVE, relative_path="/files")
    error = exc_info.value
    assert error.error_code == "GOOGLE_WORKSPACE_MALFORMED_RESPONSE"
    assert error.kind is GoogleWorkspaceErrorKind.MALFORMED_RESPONSE
    assert error.safe_reason == "invalid_response_headers"
    assert error.status_code == 401
    assert error.attempts == 1
    assert error.retry_after_seconds is None
    assert _SECRET_TYPED_ERROR_FRAGMENT not in error.safe_reason
    assert _SECRET_TYPED_ERROR_FRAGMENT not in str(error)
    assert _SECRET_TYPED_ERROR_FRAGMENT not in repr(error)


class _TypedErrorHeadersOnSecondAttempt:
    status_code = 200
    content = b'{"ok": true}'

    @property
    def headers(self) -> _TypedErrorRaisingItemsMapping:
        return _TypedErrorRaisingItemsMapping()

    def json(self) -> object:
        return {"ok": True}


def test_typed_response_headers_on_retry_preserves_real_status_and_attempt() -> None:
    policy = GoogleWorkspaceRetryPolicy(max_attempts=3)
    executor = _RecordingExecutor(
        responses=[
            _FakeResponse(503, content=b"{}"),
            _TypedErrorHeadersOnSecondAttempt(),  # type: ignore[list-item]
        ]
    )
    sleeps: list[float] = []
    transport = _transport(executor, policy=policy, sleeper=sleeps, jitter_values=[0.5])
    with pytest.raises(GoogleWorkspaceApiError) as exc_info:
        transport.get_json(source_kind=GoogleWorkspaceSourceKind.DRIVE, relative_path="/files")
    error = exc_info.value
    assert error.error_code == "GOOGLE_WORKSPACE_MALFORMED_RESPONSE"
    assert error.safe_reason == "invalid_response_headers"
    assert error.status_code == 200
    assert error.attempts == 2
    assert error.retry_after_seconds is None
    assert len(executor.calls) == 2
    assert len(sleeps) == 1
    assert _SECRET_TYPED_ERROR_FRAGMENT not in error.safe_reason
    assert _SECRET_TYPED_ERROR_FRAGMENT not in str(error)
    assert _SECRET_TYPED_ERROR_FRAGMENT not in repr(error)


@pytest.mark.parametrize("accept_header", ["Accept", "accept", "ACCEPT"])
def test_caller_accept_header_rejected_before_executor(accept_header: str) -> None:
    executor = _RecordingExecutor()
    transport = _transport(executor)
    with pytest.raises(GoogleWorkspaceApiError) as exc_info:
        transport.get_json(
            source_kind=GoogleWorkspaceSourceKind.DRIVE,
            relative_path="/files",
            headers={accept_header: "text/plain"},
        )
    assert exc_info.value.safe_reason == "forbidden_header"
    assert executor.calls == []


def test_executor_receives_exactly_one_canonical_accept_header() -> None:
    executor = _RecordingExecutor(responses=[_FakeResponse(200, content=b'{"ok": true}')])
    transport = _transport(executor)
    headers = {"X-Custom": "safe", "X-Request-Id": "trace-1"}
    transport.get_json(
        source_kind=GoogleWorkspaceSourceKind.DRIVE,
        relative_path="/files",
        headers=headers,
    )
    sent_headers = executor.calls[0]["headers"]
    assert sent_headers["Accept"] == "application/json"
    accept_keys = [name for name in sent_headers if name.casefold() == "accept"]
    assert accept_keys == ["Accept"]
    assert sent_headers["X-Custom"] == "safe"
    assert sent_headers["X-Request-Id"] == "trace-1"
    assert headers == {"X-Custom": "safe", "X-Request-Id": "trace-1"}


# --- Binary transport ---


def test_binary_get_exact_url_params_and_accept() -> None:
    executor = _RecordingExecutor(
        responses=[_FakeResponse(200, headers={"Content-Type": "application/pdf"}, content=b"%PDF")]
    )
    transport = _transport(executor)
    result = transport.get_bytes(
        source_kind=GoogleWorkspaceSourceKind.DRIVE,
        relative_path="/files/file-1",
        params={"alt": "media", "supportsAllDrives": True},
        expected_content_type="application/pdf",
        max_bytes=1024,
        range_limited=False,
    )
    assert result.data == b"%PDF"
    assert result.content_type == "application/pdf"
    call = executor.calls[0]
    assert call["url"] == "https://www.googleapis.com/drive/v3/files/file-1"
    assert call["params"] == {"alt": "media", "supportsAllDrives": True}
    assert call["headers"]["Accept"] == "application/pdf"
    assert "Range" not in call["headers"]
    assert "Authorization" not in call["headers"]


def test_binary_range_mode_sends_exact_range_header() -> None:
    executor = _RecordingExecutor(
        responses=[_FakeResponse(200, headers={"Content-Type": "text/plain"}, content=b"ok")]
    )
    transport = _transport(executor)
    transport.get_bytes(
        source_kind=GoogleWorkspaceSourceKind.DRIVE,
        relative_path="/files/file-1",
        params=None,
        expected_content_type="text/plain",
        max_bytes=4096,
        range_limited=True,
    )
    assert executor.calls[0]["headers"]["Range"] == "bytes=0-4096"


def test_binary_payload_bytes_hidden_from_repr() -> None:
    payload = GoogleWorkspaceBinaryPayload(data=b"secret", content_type="text/plain")
    rendered = repr(payload)
    assert "secret" not in rendered
    assert "data=" not in rendered


def test_binary_content_type_with_charset_accepted() -> None:
    executor = _RecordingExecutor(
        responses=[
            _FakeResponse(
                200,
                headers={"Content-Type": "application/pdf; charset=binary"},
                content=b"x",
            )
        ]
    )
    transport = _transport(executor)
    result = transport.get_bytes(
        source_kind=GoogleWorkspaceSourceKind.DRIVE,
        relative_path="/files/file-1",
        params=None,
        expected_content_type="application/pdf",
        max_bytes=10,
        range_limited=False,
    )
    assert result.content_type == "application/pdf"


def test_binary_optional_content_length_validated() -> None:
    executor = _RecordingExecutor(
        responses=[
            _FakeResponse(
                200,
                headers={"Content-Type": "text/plain", "Content-Length": "3"},
                content=b"abc",
            )
        ]
    )
    transport = _transport(executor)
    transport.get_bytes(
        source_kind=GoogleWorkspaceSourceKind.DRIVE,
        relative_path="/files/file-1",
        params=None,
        expected_content_type="text/plain",
        max_bytes=10,
        range_limited=False,
    )


def test_binary_valid_206_with_content_range() -> None:
    data = b"12345"
    executor = _RecordingExecutor(
        responses=[
            _FakeResponse(
                206,
                headers={
                    "Content-Type": "text/plain",
                    "Content-Range": "bytes 0-4/5",
                },
                content=data,
            )
        ]
    )
    transport = _transport(executor)
    result = transport.get_bytes(
        source_kind=GoogleWorkspaceSourceKind.DRIVE,
        relative_path="/files/file-1",
        params=None,
        expected_content_type="text/plain",
        max_bytes=10,
        range_limited=True,
    )
    assert result.data == data


@pytest.mark.parametrize(
    ("content_range", "safe_reason"),
    [
        ("bytes 1-4/5", "invalid_content_range"),
        ("bytes 0-3/5", "invalid_content_range"),
        ("bytes 0-4/*", "invalid_content_range"),
        (None, "invalid_content_range"),
    ],
)
def test_binary_malformed_content_range(content_range: str | None, safe_reason: str) -> None:
    headers: dict[str, str] = {"Content-Type": "text/plain"}
    if content_range is not None:
        headers["Content-Range"] = content_range
    executor = _RecordingExecutor(responses=[_FakeResponse(206, headers=headers, content=b"12345")])
    transport = _transport(executor)
    with pytest.raises(GoogleWorkspaceApiError) as exc_info:
        transport.get_bytes(
            source_kind=GoogleWorkspaceSourceKind.DRIVE,
            relative_path="/files/file-1",
            params=None,
            expected_content_type="text/plain",
            max_bytes=10,
            range_limited=True,
        )
    assert exc_info.value.safe_reason == safe_reason


def test_binary_206_total_over_limit_raises_payload_too_large() -> None:
    executor = _RecordingExecutor(
        responses=[
            _FakeResponse(
                206,
                headers={"Content-Type": "text/plain", "Content-Range": "bytes 0-4/20"},
                content=b"12345",
            )
        ]
    )
    transport = _transport(executor)
    with pytest.raises(GoogleWorkspaceApiError) as exc_info:
        transport.get_bytes(
            source_kind=GoogleWorkspaceSourceKind.DRIVE,
            relative_path="/files/file-1",
            params=None,
            expected_content_type="text/plain",
            max_bytes=10,
            range_limited=True,
        )
    assert exc_info.value.kind is GoogleWorkspaceErrorKind.PAYLOAD_TOO_LARGE


def test_binary_body_over_limit_raises_payload_too_large() -> None:
    executor = _RecordingExecutor(
        responses=[
            _FakeResponse(
                200,
                headers={"Content-Type": "text/plain"},
                content=b"x" * 11,
            )
        ]
    )
    transport = _transport(executor)
    with pytest.raises(GoogleWorkspaceApiError) as exc_info:
        transport.get_bytes(
            source_kind=GoogleWorkspaceSourceKind.DRIVE,
            relative_path="/files/file-1",
            params=None,
            expected_content_type="text/plain",
            max_bytes=10,
            range_limited=False,
        )
    assert exc_info.value.kind is GoogleWorkspaceErrorKind.PAYLOAD_TOO_LARGE


def test_binary_206_in_non_range_mode_rejected() -> None:
    executor = _RecordingExecutor(
        responses=[
            _FakeResponse(
                206,
                headers={"Content-Type": "text/plain", "Content-Range": "bytes 0-0/1"},
                content=b"x",
            )
        ]
    )
    transport = _transport(executor)
    with pytest.raises(GoogleWorkspaceApiError) as exc_info:
        transport.get_bytes(
            source_kind=GoogleWorkspaceSourceKind.DRIVE,
            relative_path="/files/file-1",
            params=None,
            expected_content_type="text/plain",
            max_bytes=10,
            range_limited=False,
        )
    assert exc_info.value.safe_reason == "unexpected_partial_content"


def test_binary_unexpected_2xx_status_rejected() -> None:
    executor = _RecordingExecutor(
        responses=[_FakeResponse(204, headers={"Content-Type": "text/plain"}, content=b"")]
    )
    transport = _transport(executor)
    with pytest.raises(GoogleWorkspaceApiError) as exc_info:
        transport.get_bytes(
            source_kind=GoogleWorkspaceSourceKind.DRIVE,
            relative_path="/files/file-1",
            params=None,
            expected_content_type="text/plain",
            max_bytes=10,
            range_limited=False,
        )
    assert exc_info.value.safe_reason == "unexpected_binary_status"


def test_binary_missing_content_type_rejected() -> None:
    executor = _RecordingExecutor(responses=[_FakeResponse(200, content=b"x")])
    transport = _transport(executor)
    with pytest.raises(GoogleWorkspaceApiError) as exc_info:
        transport.get_bytes(
            source_kind=GoogleWorkspaceSourceKind.DRIVE,
            relative_path="/files/file-1",
            params=None,
            expected_content_type="text/plain",
            max_bytes=10,
            range_limited=False,
        )
    assert exc_info.value.safe_reason == "invalid_content_type"


def test_binary_mismatched_content_type_rejected() -> None:
    executor = _RecordingExecutor(
        responses=[_FakeResponse(200, headers={"Content-Type": "application/pdf"}, content=b"x")]
    )
    transport = _transport(executor)
    with pytest.raises(GoogleWorkspaceApiError) as exc_info:
        transport.get_bytes(
            source_kind=GoogleWorkspaceSourceKind.DRIVE,
            relative_path="/files/file-1",
            params=None,
            expected_content_type="text/plain",
            max_bytes=10,
            range_limited=False,
        )
    assert exc_info.value.safe_reason == "invalid_content_type"


def test_binary_duplicate_differently_cased_content_type_rejected() -> None:
    class _DupHeaders(Mapping[str, str]):
        def __getitem__(self, key: str) -> str:
            return "text/plain"

        def __iter__(self):
            yield from ("Content-Type", "content-type")

        def __len__(self) -> int:
            return 2

    class _DupResponse:
        status_code = 200
        headers = _DupHeaders()
        content = b"x"

        def json(self) -> object:
            return {}

    executor = _RecordingExecutor()
    executor.responses.append(_DupResponse())  # type: ignore[arg-type]
    transport = _transport(executor)
    with pytest.raises(GoogleWorkspaceApiError) as exc_info:
        transport.get_bytes(
            source_kind=GoogleWorkspaceSourceKind.DRIVE,
            relative_path="/files/file-1",
            params=None,
            expected_content_type="text/plain",
            max_bytes=10,
            range_limited=False,
        )
    assert exc_info.value.safe_reason == "invalid_content_type"


def test_binary_malformed_content_length_rejected() -> None:
    executor = _RecordingExecutor(
        responses=[
            _FakeResponse(
                200,
                headers={"Content-Type": "text/plain", "Content-Length": "abc"},
                content=b"x",
            )
        ]
    )
    transport = _transport(executor)
    with pytest.raises(GoogleWorkspaceApiError) as exc_info:
        transport.get_bytes(
            source_kind=GoogleWorkspaceSourceKind.DRIVE,
            relative_path="/files/file-1",
            params=None,
            expected_content_type="text/plain",
            max_bytes=10,
            range_limited=False,
        )
    assert exc_info.value.safe_reason == "invalid_content_length"


def test_binary_content_length_mismatch_rejected() -> None:
    executor = _RecordingExecutor(
        responses=[
            _FakeResponse(
                200,
                headers={"Content-Type": "text/plain", "Content-Length": "5"},
                content=b"abc",
            )
        ]
    )
    transport = _transport(executor)
    with pytest.raises(GoogleWorkspaceApiError) as exc_info:
        transport.get_bytes(
            source_kind=GoogleWorkspaceSourceKind.DRIVE,
            relative_path="/files/file-1",
            params=None,
            expected_content_type="text/plain",
            max_bytes=10,
            range_limited=False,
        )
    assert exc_info.value.safe_reason == "invalid_content_length"


@pytest.mark.parametrize(
    ("expected_content_type", "max_bytes", "range_limited", "safe_reason"),
    [
        ("", 10, False, "invalid_expected_content_type"),
        ("text/plain,application/json", 10, False, "invalid_expected_content_type"),
        ("text/*", 10, False, "invalid_expected_content_type"),
        ("text/plain; charset=utf-8", 10, False, "invalid_expected_content_type"),
        (True, 10, False, "invalid_expected_content_type"),  # type: ignore[list-item]
        ("text/plain", True, False, "invalid_max_bytes"),  # type: ignore[list-item]
        ("text/plain", 0, False, "invalid_max_bytes"),
        ("text/plain", 104857601, False, "invalid_max_bytes"),
        ("text/plain", 10, "yes", "invalid_range_mode"),  # type: ignore[list-item]
    ],
)
def test_binary_input_validation_rejects_before_executor(
    expected_content_type: object,
    max_bytes: object,
    range_limited: object,
    safe_reason: str,
) -> None:
    executor = _RecordingExecutor()
    transport = _transport(executor)
    with pytest.raises(GoogleWorkspaceApiError) as exc_info:
        transport.get_bytes(
            source_kind=GoogleWorkspaceSourceKind.DRIVE,
            relative_path="/files/file-1",
            params=None,
            expected_content_type=expected_content_type,  # type: ignore[arg-type]
            max_bytes=max_bytes,  # type: ignore[arg-type]
            range_limited=range_limited,  # type: ignore[arg-type]
        )
    assert exc_info.value.safe_reason == safe_reason
    assert exc_info.value.attempts == 0
    assert executor.calls == []


def test_binary_oversized_error_body_uses_json_limit_not_binary_limit() -> None:
    policy = GoogleWorkspaceRetryPolicy(max_response_bytes=8, max_attempts=1)
    executor = _RecordingExecutor(responses=[_FakeResponse(500, content=b"123456789")])
    transport = _transport(executor, policy=policy)
    with pytest.raises(GoogleWorkspaceApiError) as exc_info:
        transport.get_bytes(
            source_kind=GoogleWorkspaceSourceKind.DRIVE,
            relative_path="/files/file-1",
            params=None,
            expected_content_type="text/plain",
            max_bytes=1_000_000,
            range_limited=False,
        )
    assert exc_info.value.safe_reason == "response_too_large"
    assert exc_info.value.kind is GoogleWorkspaceErrorKind.MALFORMED_RESPONSE


def test_binary_retry_on_429_and_successful_second_attempt() -> None:
    policy = GoogleWorkspaceRetryPolicy(max_attempts=3)
    executor = _RecordingExecutor(
        responses=[
            _FakeResponse(429, content=b"{}", headers={"Retry-After": "1"}),
            _FakeResponse(200, headers={"Content-Type": "text/plain"}, content=b"ok"),
        ]
    )
    sleeps: list[float] = []
    transport = _transport(executor, policy=policy, sleeper=sleeps)
    result = transport.get_bytes(
        source_kind=GoogleWorkspaceSourceKind.DRIVE,
        relative_path="/files/file-1",
        params=None,
        expected_content_type="text/plain",
        max_bytes=10,
        range_limited=False,
    )
    assert result.data == b"ok"
    assert len(executor.calls) == 2
    assert sleeps == [1.0]


def test_binary_executor_exception_maps_to_temporary() -> None:
    executor = _RecordingExecutor(errors=[RuntimeError("network secret")])
    transport = _transport(executor)
    with pytest.raises(GoogleWorkspaceApiError) as exc_info:
        transport.get_bytes(
            source_kind=GoogleWorkspaceSourceKind.DRIVE,
            relative_path="/files/file-1",
            params=None,
            expected_content_type="text/plain",
            max_bytes=10,
            range_limited=False,
        )
    assert exc_info.value.kind is GoogleWorkspaceErrorKind.TEMPORARY
    assert "network secret" not in str(exc_info.value)


# --- Binary payload self-validation ---


def test_binary_payload_valid_construction() -> None:
    payload = GoogleWorkspaceBinaryPayload(data=b"abc", content_type="application/pdf")
    assert payload.data == b"abc"
    assert payload.content_type == "application/pdf"


def test_binary_payload_uppercase_content_type_canonicalized() -> None:
    payload = GoogleWorkspaceBinaryPayload(data=b"x", content_type="APPLICATION/PDF")
    assert payload.content_type == "application/pdf"


def test_binary_payload_str_data_rejected() -> None:
    with pytest.raises(TypeError):
        GoogleWorkspaceBinaryPayload(data="abc", content_type="text/plain")  # type: ignore[arg-type]


def test_binary_payload_bytearray_rejected() -> None:
    with pytest.raises(TypeError):
        GoogleWorkspaceBinaryPayload(data=bytearray(b"abc"), content_type="text/plain")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "content_type",
    [
        "",
        "text/plain; charset=utf-8",
        "text/*",
        "text/plain,application/json",
        " text/plain",
        "text/plain ",
        "text/\x00plain",
    ],
)
def test_binary_payload_invalid_content_type_rejected(content_type: str) -> None:
    with pytest.raises(ValueError):
        GoogleWorkspaceBinaryPayload(data=b"x", content_type=content_type)


def test_binary_quoted_charset_accepted() -> None:
    executor = _RecordingExecutor(
        responses=[
            _FakeResponse(
                200,
                headers={"Content-Type": 'text/plain; charset="utf-8"'},
                content=b"ok",
            )
        ]
    )
    transport = _transport(executor)
    result = transport.get_bytes(
        source_kind=GoogleWorkspaceSourceKind.DRIVE,
        relative_path="/files/file-1",
        params=None,
        expected_content_type="text/plain",
        max_bytes=10,
        range_limited=False,
    )
    assert result.data == b"ok"


@pytest.mark.parametrize(
    ("content_type", "content"),
    [
        ("text/plain;", b"x"),
        ("text/plain; broken", b"x"),
        ("text/plain; =value", b"x"),
        ("text/plain; charset=", b"x"),
        ('text/plain; charset="unterminated', b"x"),
        ("text/plain;\r\nInjected: value", b"x"),
    ],
)
def test_binary_malformed_content_type_parameters_rejected(
    content_type: str,
    content: bytes,
) -> None:
    executor = _RecordingExecutor(
        responses=[_FakeResponse(200, headers={"Content-Type": content_type}, content=content)]
    )
    transport = _transport(executor)
    with pytest.raises(GoogleWorkspaceApiError) as exc_info:
        transport.get_bytes(
            source_kind=GoogleWorkspaceSourceKind.DRIVE,
            relative_path="/files/file-1",
            params=None,
            expected_content_type="text/plain",
            max_bytes=10,
            range_limited=False,
        )
    assert exc_info.value.safe_reason == "invalid_content_type"


def test_binary_206_incomplete_total_within_limit_rejected() -> None:
    executor = _RecordingExecutor(
        responses=[
            _FakeResponse(
                206,
                headers={"Content-Type": "text/plain", "Content-Range": "bytes 0-4/10"},
                content=b"12345",
            )
        ]
    )
    transport = _transport(executor)
    with pytest.raises(GoogleWorkspaceApiError) as exc_info:
        transport.get_bytes(
            source_kind=GoogleWorkspaceSourceKind.DRIVE,
            relative_path="/files/file-1",
            params=None,
            expected_content_type="text/plain",
            max_bytes=10,
            range_limited=True,
        )
    assert exc_info.value.safe_reason == "invalid_content_range"
