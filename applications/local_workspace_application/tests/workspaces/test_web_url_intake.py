# © Artur Czarnecki. All rights reserved.

"""WEB_URL locator model, repository and intake acceptance tests."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.queueing.providers.document_store import DocumentStoreTaskQueue
from intergrax.tools.registry.wiring import ToolWiringContext
from intergrax.websearch.capture.contracts import WebContentCaptureError, WebContentCaptureErrorCode
from intergrax.websearch.capture.url_policy import WebUrlAccessPolicy
from local_workspace_application.workspaces.knowledge_intake import KnowledgeIntakeService
from local_workspace_application.workspaces.models import (
    KnowledgeInputKind,
    WebUrlSourceLocator,
    Workspace,
    WorkspaceStatus,
)
from local_workspace_application.workspaces.repository import ManagedWorkspaceRepository
from local_workspace_application.workspaces.service import ManagedWorkspaceService
from local_workspace_application.workspaces.web_url_ingestion import (
    WebUrlAlreadyRegistered,
    WebUrlIdempotencyConflict,
    WebUrlIntakeService,
    WebUrlStateConflict,
    WebUrlValidationError,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

TENANT = "tenant-a"
WORKSPACE = "workspace-a"
PUBLIC_IP = "93.184.216.34"


async def _resolve_public(_host: str) -> tuple[str, ...]:
    return (PUBLIC_IP,)


async def _resolve_private(_host: str) -> tuple[str, ...]:
    return ("10.0.0.1",)


async def _resolve_mixed(_host: str) -> tuple[str, ...]:
    return (PUBLIC_IP, "10.0.0.1")


async def _resolve_fail(_host: str) -> tuple[str, ...]:
    raise WebContentCaptureError(WebContentCaptureErrorCode.WEB_URL_RESOLUTION_FAILED)


def _now() -> datetime:
    return datetime.now(UTC)


def _seed_workspace(repo: ManagedWorkspaceRepository) -> Workspace:
    workspace = Workspace(
        workspace_id=WORKSPACE,
        tenant_id=TENANT,
        name="Demo",
        status=WorkspaceStatus.ACTIVE,
        created_at=_now(),
        updated_at=_now(),
    )
    return repo.put_workspace(workspace)


def _locator(*, input_id: str, fingerprint: str, url: str = "https://example.com/docs") -> WebUrlSourceLocator:
    now = _now()
    return WebUrlSourceLocator(
        tenant_id=TENANT,
        workspace_id=WORKSPACE,
        input_id=input_id,
        canonical_private_url=url,
        requested_url_fingerprint=fingerprint,
        safe_display_url="https://example.com/docs",
        created_at=now,
        updated_at=now,
    )


def _build_intake(
    *,
    dns_resolver=_resolve_public,
    repository: ManagedWorkspaceRepository | None = None,
) -> tuple[ManagedWorkspaceRepository, WebUrlIntakeService]:
    store = InMemoryDocumentStore()
    repo = repository or ManagedWorkspaceRepository(store)
    policy = WebUrlAccessPolicy(dns_resolver=dns_resolver)
    queue = DocumentStoreTaskQueue(store)
    from local_workspace_application.workspaces.web_url_ingestion import WebUrlSourceResolver

    intake = KnowledgeIntakeService(
        repo,
        WebUrlSourceResolver(repo),
        ToolWiringContext(message_bus=queue),
    )
    service = WebUrlIntakeService(repo, intake, policy)
    if repository is None:
        _seed_workspace(repo)
    return repo, service


@pytest.mark.asyncio
async def test_locator_round_trip_and_repr() -> None:
    repo, _ = _build_intake()
    fingerprint = "sha256:" + "a" * 64
    locator = _locator(input_id="ki:1", fingerprint=fingerprint, url="https://example.com/x?q=1")
    repo.put_web_url_locator(locator)
    loaded = repo.get_web_url_locator(
        tenant_id=TENANT,
        workspace_id=WORKSPACE,
        requested_url_fingerprint=fingerprint,
    )
    assert loaded is not None
    assert loaded.canonical_private_url == "https://example.com/x?q=1"
    text = repr(loaded)
    assert "canonical_private_url" not in text
    assert "?q=1" not in text
    assert fingerprint in text


@pytest.mark.asyncio
async def test_locator_tenant_workspace_isolation() -> None:
    repo, _ = _build_intake()
    fingerprint = "sha256:" + "b" * 64
    repo.put_web_url_locator(_locator(input_id="ki:1", fingerprint=fingerprint))
    assert (
        repo.get_web_url_locator(
            tenant_id="other",
            workspace_id=WORKSPACE,
            requested_url_fingerprint=fingerprint,
        )
        is None
    )


@pytest.mark.asyncio
async def test_workspace_delete_removes_locators() -> None:
    repo, _ = _build_intake()
    fingerprint = "sha256:" + "c" * 64
    repo.put_web_url_locator(_locator(input_id="ki:1", fingerprint=fingerprint))
    service = ManagedWorkspaceService(repo)
    assert service.delete_workspace(tenant_id=TENANT, workspace_id=WORKSPACE) is True
    assert repo.get_web_url_locator(
        tenant_id=TENANT,
        workspace_id=WORKSPACE,
        requested_url_fingerprint=fingerprint,
    ) is None


@pytest.mark.asyncio
async def test_accept_https_url_with_query_and_fragment() -> None:
    repo, intake = _build_intake()
    accepted = await intake.accept(
        tenant_id=TENANT,
        workspace_id=WORKSPACE,
        raw_url="https://example.com/docs?language=pl#section",
        idempotency_key="url-1",
    )
    assert accepted.safe_display_url == "https://example.com/docs"
    locator = repo.get_web_url_locator(
        tenant_id=TENANT,
        workspace_id=WORKSPACE,
        requested_url_fingerprint=repo.list_knowledge_inputs(tenant_id=TENANT, workspace_id=WORKSPACE)[
            0
        ].submission_metadata["source_fingerprint"],
    )
    assert locator is not None
    assert "?language=pl" in locator.canonical_private_url
    assert "#section" not in locator.canonical_private_url
    ki = repo.list_knowledge_inputs(tenant_id=TENANT, workspace_id=WORKSPACE)[0]
    assert ki.input_kind is KnowledgeInputKind.WEB_URL
    assert set(ki.submission_metadata.keys()) == {"source_fingerprint"}
    assert "https://" not in ki.submission_metadata["source_fingerprint"]


@pytest.mark.asyncio
async def test_idempotent_same_key_same_url() -> None:
    _, intake = _build_intake()
    first = await intake.accept(
        tenant_id=TENANT,
        workspace_id=WORKSPACE,
        raw_url="https://example.com/a",
        idempotency_key="idem-1",
    )
    second = await intake.accept(
        tenant_id=TENANT,
        workspace_id=WORKSPACE,
        raw_url="https://example.com/a",
        idempotency_key="idem-1",
    )
    assert first.input_id == second.input_id
    assert first.source_id == second.source_id
    assert first.operation_id == second.operation_id


@pytest.mark.asyncio
async def test_idempotency_conflict_same_key_different_url() -> None:
    _, intake = _build_intake()
    await intake.accept(
        tenant_id=TENANT,
        workspace_id=WORKSPACE,
        raw_url="https://example.com/a",
        idempotency_key="idem-2",
    )
    with pytest.raises(WebUrlIdempotencyConflict):
        await intake.accept(
            tenant_id=TENANT,
            workspace_id=WORKSPACE,
            raw_url="https://example.com/b",
            idempotency_key="idem-2",
        )


@pytest.mark.asyncio
async def test_already_registered_different_key_same_url() -> None:
    _, intake = _build_intake()
    await intake.accept(
        tenant_id=TENANT,
        workspace_id=WORKSPACE,
        raw_url="https://example.com/shared",
        idempotency_key="key-a",
    )
    with pytest.raises(WebUrlAlreadyRegistered):
        await intake.accept(
            tenant_id=TENANT,
            workspace_id=WORKSPACE,
            raw_url="https://example.com/shared",
            idempotency_key="key-b",
        )


@pytest.mark.asyncio
async def test_same_url_allowed_in_other_workspace() -> None:
    repo, intake = _build_intake()
    other = Workspace(
        workspace_id="workspace-b",
        tenant_id=TENANT,
        name="Other",
        status=WorkspaceStatus.ACTIVE,
        created_at=_now(),
        updated_at=_now(),
    )
    repo.put_workspace(other)
    await intake.accept(
        tenant_id=TENANT,
        workspace_id=WORKSPACE,
        raw_url="https://example.com/shared",
        idempotency_key="w1",
    )
    second = await intake.accept(
        tenant_id=TENANT,
        workspace_id="workspace-b",
        raw_url="https://example.com/shared",
        idempotency_key="w2",
    )
    assert second.workspace_id == "workspace-b"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("url", "resolver", "code"),
    [
        ("http://example.com", _resolve_public, "web_url_scheme_not_allowed"),
        ("https://user:pass@example.com", _resolve_public, "web_url_credentials_not_allowed"),
        ("https://example.com:8080", _resolve_public, "web_url_port_not_allowed"),
        ("https://localhost/docs", _resolve_public, "web_url_host_not_allowed"),
        ("https://host.local/docs", _resolve_public, "web_url_host_not_allowed"),
        ("https://host.internal/docs", _resolve_public, "web_url_host_not_allowed"),
        ("https://127.0.0.1/docs", _resolve_public, "web_url_host_not_allowed"),
        ("https://example.com/docs", _resolve_private, "web_url_non_global_address_blocked"),
        ("https://example.com/docs", _resolve_mixed, "web_url_non_global_address_blocked"),
        ("https://example.com/docs", _resolve_fail, "web_url_resolution_failed"),
    ],
)
async def test_policy_rejections(url: str, resolver, code: str) -> None:
    _, intake = _build_intake(dns_resolver=resolver)
    with pytest.raises(WebUrlValidationError) as exc:
        await intake.accept(
            tenant_id=TENANT,
            workspace_id=WORKSPACE,
            raw_url=url,
            idempotency_key=f"reject-{code}",
        )
    assert exc.value.error_code == code


@pytest.mark.asyncio
async def test_cross_tenant_workspace_not_found() -> None:
    _, intake = _build_intake()
    with pytest.raises(LookupError):
        await intake.accept(
            tenant_id="other-tenant",
            workspace_id=WORKSPACE,
            raw_url="https://example.com/a",
            idempotency_key="nf",
        )


class _SpyDnsResolver:
    def __init__(self, inner) -> None:
        self._inner = inner
        self.calls: list[str] = []

    async def __call__(self, host: str) -> tuple[str, ...]:
        self.calls.append(host)
        return await self._inner(host)


@pytest.mark.asyncio
async def test_idempotent_retry_skips_dns_preflight() -> None:
    spy = _SpyDnsResolver(_resolve_public)
    repo, intake = _build_intake(dns_resolver=spy)
    first = await intake.accept(
        tenant_id=TENANT,
        workspace_id=WORKSPACE,
        raw_url="https://example.com/resume",
        idempotency_key="dns-retry",
    )
    first_call_count = len(spy.calls)
    assert first_call_count >= 1

    failing = _SpyDnsResolver(_resolve_fail)
    _, retry_intake = _build_intake(dns_resolver=failing, repository=repo)
    second = await retry_intake.accept(
        tenant_id=TENANT,
        workspace_id=WORKSPACE,
        raw_url="https://example.com/resume",
        idempotency_key="dns-retry",
    )
    assert failing.calls == []
    assert first.input_id == second.input_id
    assert first.source_id == second.source_id
    assert first.operation_id == second.operation_id


@pytest.mark.asyncio
async def test_new_url_still_requires_dns_preflight() -> None:
    failing = _SpyDnsResolver(_resolve_fail)
    _, intake = _build_intake(dns_resolver=failing)
    with pytest.raises(WebUrlValidationError) as exc:
        await intake.accept(
            tenant_id=TENANT,
            workspace_id=WORKSPACE,
            raw_url="https://example.com/brand-new",
            idempotency_key="dns-new",
        )
    assert exc.value.error_code == "web_url_resolution_failed"
    assert len(failing.calls) >= 1


@pytest.mark.asyncio
async def test_resume_missing_locator_fails_closed() -> None:
    repo, intake = _build_intake()
    accepted = await intake.accept(
        tenant_id=TENANT,
        workspace_id=WORKSPACE,
        raw_url="https://example.com/missing-locator",
        idempotency_key="loc-missing",
    )
    fingerprint = repo.list_knowledge_inputs(tenant_id=TENANT, workspace_id=WORKSPACE)[
        0
    ].submission_metadata["source_fingerprint"]
    locator = repo.get_web_url_locator(
        tenant_id=TENANT,
        workspace_id=WORKSPACE,
        requested_url_fingerprint=fingerprint,
    )
    assert locator is not None
    repo.delete_web_url_locator(
        tenant_id=TENANT,
        workspace_id=WORKSPACE,
        requested_url_fingerprint=fingerprint,
    )

    with pytest.raises(WebUrlStateConflict):
        await intake.accept(
            tenant_id=TENANT,
            workspace_id=WORKSPACE,
            raw_url="https://example.com/missing-locator",
            idempotency_key="loc-missing",
        )


@pytest.mark.asyncio
async def test_resume_locator_mismatch_fails_closed() -> None:
    repo, intake = _build_intake()
    await intake.accept(
        tenant_id=TENANT,
        workspace_id=WORKSPACE,
        raw_url="https://example.com/locator-mismatch",
        idempotency_key="loc-mismatch",
    )
    fingerprint = repo.list_knowledge_inputs(tenant_id=TENANT, workspace_id=WORKSPACE)[
        0
    ].submission_metadata["source_fingerprint"]
    locator = repo.get_web_url_locator(
        tenant_id=TENANT,
        workspace_id=WORKSPACE,
        requested_url_fingerprint=fingerprint,
    )
    assert locator is not None
    repo.put_web_url_locator(
        locator.model_copy(update={"input_id": "ki:wrong"})
    )

    with pytest.raises(WebUrlStateConflict):
        await intake.accept(
            tenant_id=TENANT,
            workspace_id=WORKSPACE,
            raw_url="https://example.com/locator-mismatch",
            idempotency_key="loc-mismatch",
        )
