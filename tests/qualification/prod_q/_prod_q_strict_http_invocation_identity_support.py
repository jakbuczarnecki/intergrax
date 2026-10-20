# © Artur Czarnecki. All rights reserved.

"""Shared helpers — PROD-Q-R1 strict HTTP invocation identity qualification."""

from __future__ import annotations

import sqlite3
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Final, Protocol, runtime_checkable

import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

from applications.governed_contractor_application.host.external_work_enterprise_reliability_bridge import (
    GovernedExternalWorkEnterpriseReliabilityBridge,
)
from applications.governed_contractor_application.host.production_external_work_composition import (
    build_governed_external_work_production_runtime,
)
from applications.governed_contractor_application.host.settings import GovernedContractorBackendSettings
from applications.governed_contractor_application.host.strict_production_serving import (
    create_governed_contractor_strict_production_asgi_app,
)
from applications.governed_contractor_application.tests.host.gr6_collaborative_work_test_support import (
    gr6_seeded_collaborative_work_repositories,
)
from applications.governed_contractor_application.tests.host.test_gr6_wire_production_decision_governance import (
    _test_policy_bundle,
)
from applications.governed_contractor_application.tests.host.test_gr7_a2_external_work_erl_bridge import (
    RecordingAdmissionPort,
    _policy_bundle,
)
from external_contractor_adapter.external_work_adapter import (
    META_IDEMPOTENCY_KEY,
    META_PROVIDER_ID,
    META_SCOPE_DESCRIPTION,
    META_SCOPE_DIGEST,
    META_WORKSPACE_REF,
)
from external_contractor_adapter.tests.fakes.deterministic_external_work import (
    DeterministicExternalWorkFake,
)
from governed_contractor_application.host.governed_contractor_host_runtime_composition import (
    compose_governed_contractor_host_runtime,
)
from intergrax.applications._shared.strict_production_serving_launcher import (
    StrictProductionServingLauncherError,
    StrictProductionServingSession,
)
from intergrax.contracts.external_work_provider_capabilities import (
    quote_first_partner_capability_fixture,
)
from intergrax.contracts.actor_identity import ActorIdentity, ActorKind
from intergrax.contracts.external_work import QuoteAcceptanceEvidence
from intergrax.contracts.money import MoneyAmount
from intergrax.contracts.provider_invocation import ProviderInvocation
from intergrax.contracts.sanctioned_external_work_operation import (
    META_PRIOR_EXECUTION_ID,
    META_SANCTIONED_MUTATION_OPERATION,
    SanctionedExternalWorkMutationOperation,
)
from external_contractor_adapter.external_work_adapter import (
    META_ACCEPTANCE_IDEMPOTENCY_KEY,
    META_QUOTE_ACCEPTANCE,
)
from intergrax.runtime.enterprise_reliability.sqlite_provider_invocation_store import (
    SQLiteProviderInvocationStore,
)
from testing_support.host_fixture_wiring import (
    install_diagnostic_cursor_secret,
    install_host_llm_stub,
)
from tests.qualification.prod_q._prod_q_r1_r7_q_p2_strict_integrated_e2e import load_d4_launcher_module
from tests.qualification.prod_q._prod_q_strict_http_caller_auth import (
    strict_http_auth_headers,
    with_strict_http_api_keys,
)
from tests.qualification.prod_q._prod_q_strict_external_work_provider_qualification_config import (
    with_qualified_strict_external_work_provider_pin,
)
from tests.qualification.prod_q.test_prod_q_r1_r7_q_p2_c_http_execution_admission import (
    _RUN_ROUTE_SUFFIX,
    _gr6_production_policy_bundle,
    _patch_gca_strict_http_wiring,
    _strict_http_environment_profile,
)
from intergrax.runtime.task.active_task_registry import ActiveTaskRegistryTaskScopeResolver
from tests.unit.runtime.governance.gr3_test_support import (
    StaticActiveTaskScope,
    bound_gr3_active_execution,
    default_gr3_identity_bundle,
)

_QUAL_TENANT: Final = "http-inv-qual-tenant"
_QUAL_WORKSPACE: Final = "http-inv-qual-ws"
_QUAL_PRINCIPAL: Final = "http-inv-qual-principal"
_QUAL_PROVIDER: Final = "gec3_deterministic_fake"
_QUAL_SCOPE_DIGEST: Final = "sha256:" + ("aa" * 32)
_QUAL_T0: Final = datetime(2026, 10, 16, 12, 0, 0, tzinfo=timezone.utc)


def _qual_strict_http_policy_bundle() -> object:
    from applications.governed_contractor_application.tests.host.test_gr6_wire_production_decision_governance import (
        _test_policy_bundle,
    )
    from external_contractor_adapter.side_effect_actions import ACTION_CANCEL_EXTERNAL_WORK
    from intergrax.contracts.runtime_policy_bundle import (
        ImmutableRuntimePolicyBundle,
        PolicyBundleRule,
    )
    from intergrax.contracts.runtime_policy_bundle import build_immutable_runtime_policy_bundle

    base = _test_policy_bundle()
    assert isinstance(base, ImmutableRuntimePolicyBundle)
    return build_immutable_runtime_policy_bundle(
        bundle_id=base.bundle_id,
        version=base.version,
        rules=(
            *base.rules,
            PolicyBundleRule(
                rule_id="http-inv-qual.CANCEL_EXTERNAL_WORK",
                description="allow cancel",
                effect="allow",
                match_action=ACTION_CANCEL_EXTERNAL_WORK,
            ),
        ),
        issued_at=base.issued_at,
    )


@dataclass(frozen=True, slots=True)
class StrictHttpExternalWorkCreatePrerequisite:
    execution_id: str
    quote_id: str
    scope_digest: str
    create_task_id: str
    create_run_id: str
    baseline_create_calls: int
    baseline_accept_calls: int
    baseline_cancel_calls: int
    baseline_intent_ids: tuple[str, ...]


def domain_summary_from_strict_http_run_body(body: dict[str, object]) -> dict[str, object]:
    meta = body.get("metadata")
    if not isinstance(meta, dict):
        return {}
    domain = meta.get("domain_summary") or meta.get("external_work")
    return dict(domain) if isinstance(domain, dict) else {}


def quote_acceptance_evidence_for_qual(
    quote_id: str,
    *,
    tenant_id: str,
    scope_digest: str,
) -> dict[str, object]:
    evidence = QuoteAcceptanceEvidence.model_validate(
        {
            "acceptance_id": "acc-http-inv-qual",
            "quote_id": quote_id,
            "quote_version": 1,
            "scope_digest": scope_digest,
            "actor": ActorIdentity(
                kind=ActorKind.USER,
                actor_id=_QUAL_PRINCIPAL,
                tenant_id=tenant_id,
            ),
            "accepted_at": _QUAL_T0 + timedelta(minutes=5),
            "hitl_decision_id": "hdec-http-inv-qual",
            "interrupt_id": "intr-http-inv-qual",
            "policy_decision_ref": "pol-http-inv-qual",
        },
    )
    return evidence.model_dump(mode="json")


def post_strict_http_create_prerequisite(
    client: TestClient,
    fixture: StrictHttpInvocationQualFixture,
    *,
    idempotency_key: str,
) -> StrictHttpExternalWorkCreatePrerequisite:
    baseline_create = fixture.external_work_fake.create_calls
    baseline_accept = fixture.external_work_fake.accept_calls
    baseline_cancel = fixture.external_work_fake.cancel_calls
    intents_before = list_intent_invocation_ids(fixture.provider_invocation_db)
    payload = run_request_json(fixture.tenant_id, idempotency_key=idempotency_key)
    response = client.post(
        fixture.run_url,
        json=payload,
        headers=strict_http_auth_headers(),
    )
    assert response.status_code == 200, response.text
    body = response.json()
    summary = domain_summary_from_strict_http_run_body(body)
    execution_id = str(summary.get("execution_id") or "").strip()
    quote_id = ""
    quote_raw = summary.get("quote")
    if isinstance(quote_raw, dict):
        quote_id = str(quote_raw.get("quote_id") or "").strip()
    scope_digest = str(summary.get("scope_digest") or _QUAL_SCOPE_DIGEST).strip()
    if not execution_id or not quote_id:
        execution_id, quote_id, scope_digest = _latest_create_follow_on_from_host_db(
            fixture.governed_external_work_host_db,
        )
    assert fixture.external_work_fake.create_calls == baseline_create + 1
    return StrictHttpExternalWorkCreatePrerequisite(
        execution_id=execution_id,
        quote_id=quote_id,
        scope_digest=scope_digest,
        create_task_id=str(body["task_id"]),
        create_run_id=str(body.get("run_id") or ""),
        baseline_create_calls=baseline_create,
        baseline_accept_calls=baseline_accept,
        baseline_cancel_calls=baseline_cancel,
        baseline_intent_ids=intents_before,
    )


def follow_up_strict_http_run_request_json(
    fixture: StrictHttpInvocationQualFixture,
    prerequisite: StrictHttpExternalWorkCreatePrerequisite,
    *,
    operation: SanctionedExternalWorkMutationOperation,
    idempotency_key: str,
) -> dict[str, object]:
    meta = external_work_http_metadata(
        tenant_id=fixture.tenant_id,
        idempotency_key=idempotency_key,
    )
    meta[META_SANCTIONED_MUTATION_OPERATION] = operation.value
    meta[META_PRIOR_EXECUTION_ID] = prerequisite.execution_id
    if operation is SanctionedExternalWorkMutationOperation.ACCEPT_QUOTE:
        meta[META_QUOTE_ACCEPTANCE] = quote_acceptance_evidence_for_qual(
            prerequisite.quote_id,
            tenant_id=fixture.tenant_id,
            scope_digest=prerequisite.scope_digest,
        )
        meta[META_ACCEPTANCE_IDEMPOTENCY_KEY] = idempotency_key
    return run_request_json(
        fixture.tenant_id,
        message=f"strict http external work {operation.value}",
        idempotency_key=idempotency_key,
        metadata=meta,
    )


@runtime_checkable
class _OwnedStrictHttpInvocationSessionCloseTarget(Protocol):
    """Narrow regression-only close surface — not a production session substitute."""

    def close(self) -> None: ...


_owned_strict_http_invocation_sessions: list[
    StrictProductionServingSession | _OwnedStrictHttpInvocationSessionCloseTarget
] = []
_sessions_with_completed_qualification_asgi_lifespan: set[int] = set()


def record_owned_strict_http_invocation_session_closed_via_completed_asgi_lifespan(
    session: StrictProductionServingSession,
) -> None:
    """Record successful ASGI lifespan shutdown for owned-session teardown (R09)."""
    _sessions_with_completed_qualification_asgi_lifespan.add(id(session))


def owned_strict_http_invocation_session_closed_via_completed_asgi_lifespan(
    session: StrictProductionServingSession,
) -> bool:
    """Whether qualification code recorded completed ASGI lifespan for this session."""
    return id(session) in _sessions_with_completed_qualification_asgi_lifespan


def _release_completed_qualification_asgi_lifespan_bookkeeping(
    session: StrictProductionServingSession | _OwnedStrictHttpInvocationSessionCloseTarget,
) -> None:
    if isinstance(session, StrictProductionServingSession):
        _sessions_with_completed_qualification_asgi_lifespan.discard(id(session))


def _strict_production_session_released_by_completed_qualification_asgi_lifespan(
    session: StrictProductionServingSession | _OwnedStrictHttpInvocationSessionCloseTarget,
) -> bool:
    """True when qualification recorded public session close via completed ASGI lifespan."""
    if isinstance(session, StrictProductionServingSession):
        return owned_strict_http_invocation_session_closed_via_completed_asgi_lifespan(
            session,
        )
    return False


@contextmanager
def strict_http_qualification_asgi_lifespan(
    fixture: StrictHttpInvocationQualFixture,
) -> Generator[TestClient, None, None]:
    """Run TestClient ASGI lifespan; record completion only on full successful exit."""
    body_exc: Exception | None = None
    try:
        with TestClient(fixture.app) as client:
            try:
                yield client
            except BaseException as exc:
                if not isinstance(exc, Exception):
                    raise
                body_exc = exc
    except BaseException as shutdown_exc:
        if not isinstance(shutdown_exc, Exception):
            raise
        if body_exc is not None:
            raise ExceptionGroup(
                "strict HTTP qualification ASGI lifespan exit failed after body failure",
                [body_exc, shutdown_exc],
            ) from shutdown_exc
        raise
    record_owned_strict_http_invocation_session_closed_via_completed_asgi_lifespan(
        fixture.session,
    )
    if body_exc is not None:
        raise body_exc


def register_owned_strict_http_invocation_session(
    session: StrictProductionServingSession,
) -> StrictProductionServingSession:
    """Track one strict serving session opened by qualification fixtures for teardown."""
    _owned_strict_http_invocation_sessions.append(session)
    return session


def register_owned_strict_http_invocation_close_target_for_regression(
    target: _OwnedStrictHttpInvocationSessionCloseTarget,
) -> None:
    """Register a deterministic close double for R01–R04 registry regression tests only."""
    _owned_strict_http_invocation_sessions.append(target)


def assert_no_leaked_owned_strict_http_invocation_sessions() -> None:
    if _owned_strict_http_invocation_sessions:
        raise AssertionError(
            "owned strict HTTP invocation qualification session leaked from a prior test; "
            "close via StrictProductionServingSession.close() before the next test",
        )


def owned_strict_http_invocation_session_registration_count() -> int:
    """Return how many owned sessions remain registered (qualification teardown contract)."""
    return len(_owned_strict_http_invocation_sessions)


def close_registered_owned_strict_http_invocation_sessions() -> None:
    """Close owned sessions in reverse open order using the public session contract."""
    errors: list[Exception] = []
    while _owned_strict_http_invocation_sessions:
        owned = _owned_strict_http_invocation_sessions[-1]
        if _strict_production_session_released_by_completed_qualification_asgi_lifespan(
            owned,
        ):
            _owned_strict_http_invocation_sessions.pop()
            _release_completed_qualification_asgi_lifespan_bookkeeping(owned)
            continue
        try:
            owned.close()
        except StrictProductionServingLauncherError as exc:
            errors.append(exc)
            break
        except Exception as exc:
            errors.append(exc)
            break
        else:
            _owned_strict_http_invocation_sessions.pop()
            _release_completed_qualification_asgi_lifespan_bookkeeping(owned)
    if len(errors) == 1:
        raise errors[0]
    if errors:
        first = errors[0]
        raise ExceptionGroup(
            "owned strict HTTP invocation qualification session close failed",
            errors,
        ) from first


@dataclass(frozen=True, slots=True)
class StrictHttpInvocationQualFixture:
    session: StrictProductionServingSession
    app: FastAPI
    run_url: str
    tenant_id: str
    external_work_fake: DeterministicExternalWorkFake
    provider_invocation_db: Path
    governed_external_work_host_db: Path


def _latest_create_follow_on_from_host_db(
    host_db: Path,
    *,
    known_execution_ids: frozenset[str] = frozenset(),
) -> tuple[str, str, str]:
    import json

    conn = sqlite3.connect(host_db)
    try:
        rows = conn.execute(
            "SELECT execution_id, payload FROM gca_create_follow_on ORDER BY execution_id",
        ).fetchall()
    finally:
        conn.close()
    for execution_id_raw, payload_raw in reversed(rows):
        execution_id = str(execution_id_raw)
        if known_execution_ids and execution_id in known_execution_ids:
            continue
        payload = json.loads(str(payload_raw))
        snapshot = payload.get("snapshot") or {}
        quote = snapshot.get("quote") or {}
        quote_id = str(quote.get("quote_id") or "").strip()
        scope_digest = str(quote.get("scope_digest") or snapshot.get("scope_digest") or "").strip()
        if execution_id and quote_id and scope_digest:
            return execution_id, quote_id, scope_digest
    raise AssertionError("create follow-on record with quote not found in host store")


def list_intent_invocation_ids(db_path: Path) -> tuple[str, ...]:
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            "SELECT invocation_id FROM provider_invocation_intents ORDER BY invocation_id",
        ).fetchall()
    finally:
        conn.close()
    return tuple(str(r[0]) for r in rows)


def external_work_adapter_metadata(
    *,
    tenant_id: str,
    idempotency_key: str,
    workspace_ref: str = _QUAL_WORKSPACE,
) -> dict[str, object]:
    return {
        META_PROVIDER_ID: _QUAL_PROVIDER,
        META_SCOPE_DESCRIPTION: "strict http invocation qualification scope",
        META_SCOPE_DIGEST: _QUAL_SCOPE_DIGEST,
        META_IDEMPOTENCY_KEY: idempotency_key,
        META_WORKSPACE_REF: workspace_ref,
        "external_work.budget_limit": MoneyAmount(amount=Decimal("25.00"), currency="USD"),
        "external_work.principal_id": _QUAL_PRINCIPAL,
        "external_work.tenant_id": tenant_id,
        "external_work.workspace_ref": workspace_ref,
    }


def external_work_http_metadata(
    *,
    tenant_id: str,
    idempotency_key: str,
    workspace_ref: str = _QUAL_WORKSPACE,
) -> dict[str, object]:
    """JSON-serializable metadata for strict HTTP ``/run`` bodies."""
    return {
        META_PROVIDER_ID: _QUAL_PROVIDER,
        META_SCOPE_DESCRIPTION: "strict http invocation qualification scope",
        META_SCOPE_DIGEST: _QUAL_SCOPE_DIGEST,
        META_IDEMPOTENCY_KEY: idempotency_key,
        META_WORKSPACE_REF: workspace_ref,
        "external_work.budget_limit": {"amount": "25.00", "currency": "USD"},
        "external_work.principal_id": _QUAL_PRINCIPAL,
        "external_work.tenant_id": tenant_id,
        "external_work.workspace_ref": workspace_ref,
    }


def run_request_json(
    tenant_id: str,
    *,
    message: str = "strict http external work qualification",
    idempotency_key: str = "idem-http-inv-qual-01",
    metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    from tests.qualification.prod_q._prod_q_strict_http_caller_auth import _QUAL_USER

    meta = metadata or external_work_http_metadata(
        tenant_id=tenant_id,
        idempotency_key=idempotency_key,
    )
    return {
        "tenant_id": tenant_id,
        "user_id": _QUAL_USER,
        "message": message,
        "capability": "external_contractor.adapt",
        "metadata": meta,
    }


def build_strict_http_invocation_qual_fixture(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    revision_id: str,
    default_agent_id: str = "search",
) -> StrictHttpInvocationQualFixture:
    d4, err = load_d4_launcher_module()
    assert err is None, f"integration import blocked: {err}"
    install_diagnostic_cursor_secret(monkeypatch)
    install_host_llm_stub(monkeypatch)

    external_fake = DeterministicExternalWorkFake()
    env_profile = d4._ac3_environment()
    execution_tenant = env_profile.application_environment_id
    cw = gr6_seeded_collaborative_work_repositories(
        tenant_id=execution_tenant,
        workspace_id=_QUAL_WORKSPACE,
        principal_id=_QUAL_PRINCIPAL,
    )
    http_task_scope = ActiveTaskRegistryTaskScopeResolver()

    def _host_runtime_for_startup(settings: GovernedContractorBackendSettings):
        settings = with_qualified_strict_external_work_provider_pin(
            with_strict_http_api_keys(
                replace(
                    settings,
                    include_mcp=False,
                    include_scheduler=False,
                    include_interaction_routes=False,
                    include_task_control=False,
                    runtime_policy_bundle=_qual_strict_http_policy_bundle(),  # type: ignore[arg-type]
                ),
                execution_tenant_id=execution_tenant,
            ),
        )
        from intergrax.runtime.governance.decision_requirement_policy import (
            PermissiveDecisionRequirementPolicy,
        )

        return compose_governed_contractor_host_runtime(
            settings,
            integration=external_fake,
            task_scope=http_task_scope,
            collaborative_work_repositories=cw,
            decision_requirement_policy=PermissiveDecisionRequirementPolicy(),
        )

    monkeypatch.setattr(d4, "_host_runtime_for_startup", _host_runtime_for_startup)
    _patch_gca_strict_http_wiring(monkeypatch, d4)
    for module in (
        "governed_contractor_application.host.main",
        "governed_contractor_application.host.factory",
    ):
        monkeypatch.setattr(
            f"{module}.build_governed_contractor_environment_profile",
            _strict_http_environment_profile,
        )

    session = d4._open_with_activation(
        tmp_path,
        environment=d4._ac3_environment(),
        revision_id=revision_id,
    )
    session = register_owned_strict_http_invocation_session(session)
    assert session.readiness.ready_for_traffic is True

    env = session.environment
    tenant_id = env.application_environment_id
    settings = with_qualified_strict_external_work_provider_pin(
        with_strict_http_api_keys(
            replace(
                GovernedContractorBackendSettings.from_env(),
                include_mcp=False,
                include_scheduler=False,
                include_interaction_routes=False,
                include_task_control=False,
                default_agent_id=default_agent_id,
                runtime_policy_bundle=_qual_strict_http_policy_bundle(),
            ),
            execution_tenant_id=tenant_id,
            wrong_tenant_id=f"{tenant_id}-unauthorized-caller",
        ),
    )
    host_runtime = d4._host_runtime_for_startup(settings)
    app = create_governed_contractor_strict_production_asgi_app(
        session,
        settings=settings,
        host_runtime=host_runtime,
    )
    storage, _adapter = d4._storage_and_adapter(tmp_path)
    return StrictHttpInvocationQualFixture(
        session=session,
        app=app,
        run_url=f"{settings.route_prefix}{_RUN_ROUTE_SUFFIX}",
        tenant_id=tenant_id,
        external_work_fake=external_fake,
        provider_invocation_db=storage.provider_invocation_db,
        governed_external_work_host_db=storage.governed_external_work_host_db,
    )


def build_sanctioned_external_work_runtime(
    fixture: StrictHttpInvocationQualFixture,
    *,
    identity: tuple[object, object, object, object] | None = None,
) -> tuple[object, SQLiteProviderInvocationStore, tuple[object, object, object, object]]:
    pis = fixture.session.composition.provider_invocation_store
    assert isinstance(pis, SQLiteProviderInvocationStore)
    cw = gr6_seeded_collaborative_work_repositories(
        tenant_id=fixture.tenant_id,
        workspace_id=_QUAL_WORKSPACE,
        principal_id=_QUAL_PRINCIPAL,
    )
    from governed_contractor_application.host.stores import (
        InMemoryContinuationStateStore,
        InMemoryGovernedExecutionStore,
        InMemoryPolicyBundleArtifactStore,
        InMemoryProofReceiptStore,
    )

    task_id, run_id, _attempt_id, _execution_id = (
        identity if identity is not None else default_gr3_identity_bundle()
    )
    bridge = GovernedExternalWorkEnterpriseReliabilityBridge(
        admission_port=RecordingAdmissionPort(),
    )
    production_runtime = build_governed_external_work_production_runtime(
        fixture.external_work_fake,
        tenant_id=fixture.tenant_id,
        workspace_id=_QUAL_WORKSPACE,
        principal_id=_QUAL_PRINCIPAL,
        task_scope=StaticActiveTaskScope(task_id),
        capabilities=quote_first_partner_capability_fixture(provider_id=_QUAL_PROVIDER),
        policy_bundle=_policy_bundle(),
        collaborative_work_repositories=cw,
        execution_store=InMemoryGovernedExecutionStore(),
        receipt_store=InMemoryProofReceiptStore(),
        bundle_store=InMemoryPolicyBundleArtifactStore(),
        continuation_store=InMemoryContinuationStateStore(),
        provider_invocation_store=pis,
        reliability_bridge=bridge,
        clock=lambda: _QUAL_T0,
    )
    return production_runtime, pis, (task_id, run_id, _attempt_id, _execution_id)


def fixed_provider_invocation(task_id: str, run_id: str, invocation_id: str) -> ProviderInvocation:
    return ProviderInvocation(
        invocation_id=invocation_id,
        provider_id=_QUAL_PROVIDER,
        operation="create_work",
        task_id=task_id,
        run_id=run_id,
        request_digest="sha256:" + ("bb" * 32),
        started_at=_QUAL_T0,
    )


def post_strict_http_run(
    fixture: StrictHttpInvocationQualFixture,
    payload: dict[str, object],
) -> dict[str, object]:
    with strict_http_qualification_asgi_lifespan(fixture) as client:
        response = client.post(
            fixture.run_url,
            json=payload,
            headers=strict_http_auth_headers(),
        )
    assert response.status_code == 200, response.text
    return response.json()


def execute_sanctioned_create_with_invocation(
    production_runtime: object,
    fixture: StrictHttpInvocationQualFixture,
    *,
    invocation_id: str,
    idempotency_key: str,
    identity: tuple[object, object, object, object],
) -> None:
    adapter = production_runtime.adapter
    task_id, run_id, attempt_id, execution_id = identity
    invocation = fixed_provider_invocation(str(task_id), str(run_id), invocation_id)
    request = adapter.build_create_request(
        task_id=str(task_id),
        run_id=str(run_id),
        metadata=external_work_adapter_metadata(
            tenant_id=fixture.tenant_id,
            idempotency_key=idempotency_key,
        ),
    )
    with bound_gr3_active_execution(
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
        task_id=task_id,
        governance_tenant_id=fixture.tenant_id,
        governance_workspace_id=_QUAL_WORKSPACE,
        governance_principal_id=_QUAL_PRINCIPAL,
    ):
        result = adapter.create_and_map(
            request,
            principal_id=_QUAL_PRINCIPAL,
            tenant_id=fixture.tenant_id,
            provider_invocation=invocation,
        )
    assert result.used is True, result.error_message


__all__ = [
    "StrictHttpExternalWorkCreatePrerequisite",
    "StrictHttpInvocationQualFixture",
    "assert_no_leaked_owned_strict_http_invocation_sessions",
    "close_registered_owned_strict_http_invocation_sessions",
    "owned_strict_http_invocation_session_registration_count",
    "build_sanctioned_external_work_runtime",
    "build_strict_http_invocation_qual_fixture",
    "register_owned_strict_http_invocation_close_target_for_regression",
    "register_owned_strict_http_invocation_session",
    "execute_sanctioned_create_with_invocation",
    "domain_summary_from_strict_http_run_body",
    "external_work_adapter_metadata",
    "follow_up_strict_http_run_request_json",
    "post_strict_http_create_prerequisite",
    "quote_acceptance_evidence_for_qual",
    "external_work_http_metadata",
    "list_intent_invocation_ids",
    "owned_strict_http_invocation_session_closed_via_completed_asgi_lifespan",
    "post_strict_http_run",
    "run_request_json",
    "strict_http_qualification_asgi_lifespan",
    "_QUAL_T0",
]
