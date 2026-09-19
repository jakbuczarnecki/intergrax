# © Artur Czarnecki. All rights reserved.

"""MP-6E — scoped Collaborative Activity read / query qualification."""

from __future__ import annotations

import ast
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from intergrax.collaborative_work.authority import CollaborativeWorkAuthorityResolver
from intergrax.collaborative_work.collaborative_activity_page_cursor_codec import (
    encode_collaborative_activity_page_cursor,
)
from intergrax.collaborative_work.collaborative_activity_read import CollaborativeActivityReadService
from intergrax.collaborative_work.collaborative_activity_read_authorization import (
    CollaborativeActivityReadAuthorizationEvaluator,
    DefaultCollaborativeActivityReadAuthorizationPolicy,
)
from intergrax.collaborative_work.collaborative_activity_read_store import (
    SQLiteCollaborativeActivityReadStore,
)
from dataclasses import dataclass

from intergrax.collaborative_work.in_memory_repository import (
    InMemoryAuthorityDelegationRepository,
    InMemoryPrincipalAuthorityRepository,
    InMemoryWorkspaceMembershipRepository,
)
from intergrax.collaborative_work.persistence import (
    collaborative_activity_read_store_from_sqlite_bundle,
    open_sqlite_collaborative_work_repositories,
    sqlite_collaborative_activity_append_store,
)
from intergrax.contracts.collaborative_activity import (
    CollaborativeActivityPage,
    CollaborativeActivityQuery,
    CollaborativeActivityReadPort,
)
from intergrax.contracts.collaborative_activity_read import (
    COLLABORATIVE_ACTIVITY_READ_AUTHORITY_SCOPE,
    CollaborativeActivityCursorInvalid,
    CollaborativeActivityReadAuthorizationDecision,
    CollaborativeActivityReadAuthorizationOutcome,
    CollaborativeActivityReadAuthorizationPolicy,
    CollaborativeActivityReadAuthorizationPolicyInput,
    CollaborativeActivityReadDenied,
    CollaborativeActivityReadDenialReason,
    CollaborativeActivityReadPolicyError,
    CollaborativeActivityReadRequest,
    fail_closed_collaborative_activity_read_decision,
)
from intergrax.collaborative_work.repository import (
    CreatePrincipalAuthorityGrantCommand,
    CreateWorkspaceMembershipCommand,
)
from intergrax.contracts.collaborative_work import (
    AuthorityGrantStatus,
    MembershipResolutionMode,
    MembershipStatus,
    WorkspaceMembership,
    WorkspaceMembershipRole,
)
from tests.unit.collaborative_work.collaborative_activity_append_store_contract import (
    fixed_recorded_at,
    make_intent,
    make_publication,
)
from tests.unit.collaborative_work.collaborative_activity_read_port_contract import (
    run_collaborative_activity_read_port_contract_suite,
    run_cursor_scope_mismatch_contract,
    run_invalid_cursor_contract,
    run_late_occurred_at_ordering_contract,
    run_tenant_isolation_read_contract,
    run_workspace_isolation_read_contract,
)

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[3]
_TENANT = "tenant-a"
_WORKSPACE = "ws-a"
_ACTING = "principal-reader"


@dataclass(frozen=True)
class _AuthorityFixture:
    resolver: CollaborativeWorkAuthorityResolver
    membership_repo: InMemoryWorkspaceMembershipRepository


def _open_bundle(tmp_path: Path):
    return open_sqlite_collaborative_work_repositories(str(tmp_path / "mp6e.sqlite"))


def _append_store(tmp_path: Path):
    return sqlite_collaborative_activity_append_store(
        str(tmp_path / "mp6e.sqlite"),
        utc_now=fixed_recorded_at,
    )


def _read_store(tmp_path: Path) -> SQLiteCollaborativeActivityReadStore:
    bundle = _open_bundle(tmp_path)
    return collaborative_activity_read_store_from_sqlite_bundle(bundle)


def _seed_authority(
    *,
    tenant_id: str = _TENANT,
    workspace_id: str = _WORKSPACE,
    principal_id: str = _ACTING,
    authority_scopes: tuple[str, ...] = (COLLABORATIVE_ACTIVITY_READ_AUTHORITY_SCOPE,),
) -> _AuthorityFixture:
    membership_repo = InMemoryWorkspaceMembershipRepository()
    delegation_repo = InMemoryAuthorityDelegationRepository()
    authority_repo = InMemoryPrincipalAuthorityRepository()
    membership_repo.create(
        CreateWorkspaceMembershipCommand(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            membership_id="membership-1",
            principal_id=principal_id,
            role=WorkspaceMembershipRole.MEMBER,
            status=MembershipStatus.ACTIVE,
        ),
    )
    authority_repo.create(
        CreatePrincipalAuthorityGrantCommand(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            authority_grant_id="grant-1",
            principal_id=principal_id,
            authority_scopes=authority_scopes,
            status=AuthorityGrantStatus.ACTIVE,
        ),
    )
    return _AuthorityFixture(
        resolver=CollaborativeWorkAuthorityResolver(
            membership_repository=membership_repo,
            delegation_repository=delegation_repo,
            principal_authority_repository=authority_repo,
        ),
        membership_repo=membership_repo,
    )


def _membership_locator(
    fixture: _AuthorityFixture,
    *,
    tenant_id: str = _TENANT,
    workspace_id: str = _WORKSPACE,
    principal_id: str = _ACTING,
) -> WorkspaceMembership:
    membership = fixture.membership_repo.get_for_principal(
        tenant_id=tenant_id,
        workspace_id=workspace_id,
        principal_id=principal_id,
    )
    assert membership is not None
    return membership


def _read_request(
    fixture: _AuthorityFixture,
    **query_overrides: object,
) -> CollaborativeActivityReadRequest:
    query_payload = {
        "tenant_id": _TENANT,
        "workspace_id": _WORKSPACE,
        "limit": 50,
    }
    query_payload.update(query_overrides)
    return CollaborativeActivityReadRequest(
        query=CollaborativeActivityQuery.model_validate(query_payload),
        acting_principal_id=_ACTING,
        membership_resolution_mode=MembershipResolutionMode.LOCATOR,
        membership=_membership_locator(fixture),
    )


def _read_service(
    tmp_path: Path,
    fixture: _AuthorityFixture | None = None,
    read_port: CollaborativeActivityReadPort | None = None,
    policy: CollaborativeActivityReadAuthorizationPolicy | None = None,
) -> CollaborativeActivityReadService:
    fixture = fixture or _seed_authority()
    read_port = read_port or _read_store(tmp_path)
    policy = policy or DefaultCollaborativeActivityReadAuthorizationPolicy()
    evaluator = CollaborativeActivityReadAuthorizationEvaluator(
        authority_resolver=fixture.resolver,
        read_authorization_policy=policy,
    )
    return CollaborativeActivityReadService(read_authorization=evaluator, read_port=read_port)


def test_mp6e_sqlite_read_port_contract_suite(tmp_path: Path) -> None:
    bundle = _open_bundle(tmp_path)
    append = sqlite_collaborative_activity_append_store(
        str(tmp_path / "mp6e.sqlite"),
        utc_now=fixed_recorded_at,
    )
    read_port = collaborative_activity_read_store_from_sqlite_bundle(bundle)
    run_collaborative_activity_read_port_contract_suite(
        read_port_factory=lambda: read_port,
        append_store_factory=lambda: append,
    )


def test_mp6e_workspace_and_tenant_isolation(tmp_path: Path) -> None:
    bundle = _open_bundle(tmp_path)
    append = sqlite_collaborative_activity_append_store(
        str(tmp_path / "mp6e.sqlite"),
        utc_now=fixed_recorded_at,
    )
    read_port = collaborative_activity_read_store_from_sqlite_bundle(bundle)
    run_workspace_isolation_read_contract(
        read_port_factory=lambda: read_port,
        append_store_factory=lambda: append,
    )
    run_tenant_isolation_read_contract(
        read_port_factory=lambda: read_port,
        append_store_factory=lambda: append,
    )


def test_mp6e_late_occurred_at_ordering(tmp_path: Path) -> None:
    append = _append_store(tmp_path)
    read_port = _read_store(tmp_path)
    run_late_occurred_at_ordering_contract(
        read_port_factory=lambda: read_port,
        append_store_factory=lambda: append,
    )


def test_mp6e_cursor_scope_and_invalid(tmp_path: Path) -> None:
    read_port = _read_store(tmp_path)
    run_cursor_scope_mismatch_contract(lambda: read_port)
    run_invalid_cursor_contract(lambda: read_port)


def test_mp6e_concurrent_append_between_pages(tmp_path: Path) -> None:
    append = _append_store(tmp_path)
    read_port = _read_store(tmp_path)
    for index in range(3):
        append.append_idempotent(make_intent(make_publication(stable_id=f"c-{index + 1}")))

    page1 = read_port.query(
        CollaborativeActivityQuery(tenant_id=_TENANT, workspace_id=_WORKSPACE, limit=2),
    )
    assert [item.append_position for item in page1.activities] == [1, 2]

    append.append_idempotent(make_intent(make_publication(stable_id="c-4")))

    page2 = read_port.query(
        CollaborativeActivityQuery(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            limit=2,
            cursor=page1.next_cursor,
        ),
    )
    assert [item.append_position for item in page2.activities] == [3, 4]
    assert page2.next_cursor is None

    all_positions: list[int] = [
        *(item.append_position for item in page1.activities),
        *(item.append_position for item in page2.activities),
    ]
    assert all_positions == [1, 2, 3, 4]


def test_mp6e_query_limit_bounds_rejected() -> None:
    with pytest.raises(ValueError):
        CollaborativeActivityQuery(tenant_id=_TENANT, workspace_id=_WORKSPACE, limit=0)
    with pytest.raises(ValueError):
        CollaborativeActivityQuery(tenant_id=_TENANT, workspace_id=_WORKSPACE, limit=501)


def test_mp6e_deny_zero_provider_calls(tmp_path: Path) -> None:
    spy = MagicMock(spec=CollaborativeActivityReadPort)
    fixture = _seed_authority(authority_scopes=("workspace.unrelated",))
    service = _read_service(tmp_path, fixture=fixture, read_port=spy)
    with pytest.raises(CollaborativeActivityReadDenied):
        service.read_page(_read_request(fixture))
    spy.query.assert_not_called()


def test_mp6e_policy_failure_zero_provider_calls(tmp_path: Path) -> None:
    class BrokenPolicy:
        policy_id = "broken"

        def evaluate(
            self,
            policy_input: CollaborativeActivityReadAuthorizationPolicyInput,
        ) -> CollaborativeActivityReadAuthorizationDecision:
            raise RuntimeError("policy broken")

    spy = MagicMock(spec=CollaborativeActivityReadPort)
    fixture = _seed_authority()
    service = _read_service(tmp_path, fixture=fixture, read_port=spy, policy=BrokenPolicy())
    with pytest.raises(CollaborativeActivityReadPolicyError):
        service.read_page(_read_request(fixture))
    spy.query.assert_not_called()


def test_mp6e_custom_policy_deny(tmp_path: Path) -> None:
    class DenyPolicy:
        policy_id = "deny-all"

        def evaluate(
            self,
            policy_input: CollaborativeActivityReadAuthorizationPolicyInput,
        ) -> CollaborativeActivityReadAuthorizationDecision:
            return fail_closed_collaborative_activity_read_decision(
                policy_id=self.policy_id,
                denial_reason=CollaborativeActivityReadDenialReason.POLICY_AMBIGUITY,
            )

    spy = MagicMock(spec=CollaborativeActivityReadPort)
    fixture = _seed_authority()
    service = _read_service(tmp_path, fixture=fixture, read_port=spy, policy=DenyPolicy())
    with pytest.raises(CollaborativeActivityReadDenied):
        service.read_page(_read_request(fixture))
    spy.query.assert_not_called()


def test_mp6e_custom_read_port(tmp_path: Path) -> None:
    class FakeReadPort:
        def query(self, query: CollaborativeActivityQuery) -> CollaborativeActivityPage:
            return CollaborativeActivityPage(activities=())

    fixture = _seed_authority()
    service = _read_service(tmp_path, fixture=fixture, read_port=FakeReadPort())
    page = service.read_page(_read_request(fixture))
    assert page.activities == ()


def test_mp6e_cross_tenant_request_denied_before_store(tmp_path: Path) -> None:
    spy = MagicMock(spec=CollaborativeActivityReadPort)
    fixture = _seed_authority(tenant_id=_TENANT)
    service = _read_service(tmp_path, fixture=fixture, read_port=spy)
    request = _read_request(fixture, tenant_id="tenant-b")
    with pytest.raises(CollaborativeActivityReadDenied):
        service.read_page(request)
    spy.query.assert_not_called()


def test_mp6e_invalid_cursor_zero_provider_calls(tmp_path: Path) -> None:
    spy = MagicMock(spec=CollaborativeActivityReadPort)
    fixture = _seed_authority()
    service = _read_service(tmp_path, fixture=fixture, read_port=spy)
    bad_cursor = encode_collaborative_activity_page_cursor(
        query=CollaborativeActivityQuery(
            tenant_id=_TENANT,
            workspace_id="ws-other",
            limit=10,
        ),
        after_append_position=1,
    )
    request = _read_request(fixture, cursor=bad_cursor, workspace_id=_WORKSPACE)
    with pytest.raises(CollaborativeActivityCursorInvalid):
        service.read_page(request)
    spy.query.assert_not_called()


def test_mp6e_service_authorizes_before_read(tmp_path: Path) -> None:
    append = _append_store(tmp_path)
    append.append_idempotent(make_intent(make_publication(stable_id="authorized-1")))
    fixture = _seed_authority()
    service = _read_service(tmp_path, fixture=fixture)
    page = service.read_page(_read_request(fixture))
    assert len(page.activities) == 1


def test_mp6e_read_service_has_no_provider_imports() -> None:
    module = _REPO_ROOT / "intergrax" / "collaborative_work" / "collaborative_activity_read.py"
    tree = ast.parse(module.read_text(encoding="utf-8-sig"))
    imports = {
        (node.module or "")
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
    }
    assert not any("sqlite" in item or "postgresql" in item for item in imports)


def test_mp6e_read_authorization_has_no_provider_imports() -> None:
    module = (
        _REPO_ROOT / "intergrax" / "collaborative_work" / "collaborative_activity_read_authorization.py"
    )
    text = module.read_text(encoding="utf-8-sig").lower()
    assert "sqlite" not in text
    assert "postgresql" not in text


def test_mp6e_read_contract_has_no_provider_imports() -> None:
    module = _REPO_ROOT / "intergrax" / "contracts" / "collaborative_activity_read.py"
    text = module.read_text(encoding="utf-8-sig").lower()
    for marker in ("sqlite", "postgresql", "sqlalchemy", "agents.", "applications."):
        assert marker not in text
