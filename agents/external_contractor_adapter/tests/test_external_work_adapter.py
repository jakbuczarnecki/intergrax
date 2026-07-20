# © Artur Czarnecki. All rights reserved.

"""GEC-3 — provider-neutral Tier-2 external work adapter tests."""

from __future__ import annotations

import ast
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import pytest

from external_contractor_adapter.external_contractor_adapter_agent import (
    ExternalContractorAdapterAgent,
)
from external_contractor_adapter.external_work_adapter import (
    META_ACCEPTANCE_IDEMPOTENCY_KEY,
    META_BUDGET_LIMIT,
    META_IDEMPOTENCY_KEY,
    META_PROVIDER_ID,
    META_QUOTE_ACCEPTANCE,
    META_SCOPE_DESCRIPTION,
    META_SCOPE_DIGEST,
    ExternalWorkAdapter,
    adapt_from_step_metadata,
)
from external_contractor_adapter.tests.fakes.deterministic_external_work import (
    DeterministicExternalWorkFake,
)
from external_contractor_adapter.tests.fakes.deterministic_side_effect_policy import (
    DeterministicMeaningfulSideEffectPolicy,
)
from intergrax.contracts.actor_identity import ActorIdentity, ActorKind
from intergrax.contracts.agent_run import AgentRunRequest, RequestIdentity
from intergrax.contracts.agent_run_enums import AgentRunStatus
from intergrax.contracts.external_work import (
    ExternalWorkCapability,
    ExternalWorkCreateRequest,
    ExternalWorkErrorCode,
    ExternalWorkStatus,
    QuoteAcceptanceEvidence,
)
from intergrax.contracts.money import MoneyAmount
from intergrax.contracts.runtime_policy import PolicyAction
from intergrax.integrations.contracts.external_work import (
    ExternalWorkError,
    ExternalWorkIntegration,
)

_DIGEST = "sha256:" + ("cd" * 32)
_T0 = datetime(2026, 7, 20, 14, 0, 0, tzinfo=timezone.utc)
_AGENT_ROOT = Path(__file__).resolve().parents[1]
_ADAPTER_PY = _AGENT_ROOT / "external_work_adapter.py"
_AGENT_PY = _AGENT_ROOT / "external_contractor_adapter_agent.py"
_DOMAIN_PY = _AGENT_ROOT / "steps" / "domain_job.py"


def _meta(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        META_PROVIDER_ID: "gec3_deterministic_fake",
        META_SCOPE_DESCRIPTION: "review PR #42",
        META_SCOPE_DIGEST: _DIGEST,
        META_IDEMPOTENCY_KEY: "idem-gec3-1",
        META_BUDGET_LIMIT: MoneyAmount(amount=Decimal("40.00"), currency="USD"),
    }
    payload.update(overrides)
    return payload


def _allow_policy() -> DeterministicMeaningfulSideEffectPolicy:
    return DeterministicMeaningfulSideEffectPolicy(default=PolicyAction.ALLOW)


def _adapter(
    fake: DeterministicExternalWorkFake | None = None,
    *,
    policy: DeterministicMeaningfulSideEffectPolicy | None = None,
) -> ExternalWorkAdapter:
    return ExternalWorkAdapter(
        fake or DeterministicExternalWorkFake(),
        side_effect_policy=policy or _allow_policy(),
    )


def _acceptance(**overrides: object) -> QuoteAcceptanceEvidence:
    payload: dict[str, object] = {
        "acceptance_id": "acc-gec3-1",
        "quote_id": "q-gec3-1",
        "quote_version": 1,
        "scope_digest": _DIGEST,
        "actor": ActorIdentity(
            kind=ActorKind.USER, actor_id="u1", tenant_id="tenant-a"
        ),
        "accepted_at": _T0 + timedelta(minutes=3),
        "hitl_decision_id": "hdec_gec3",
    }
    payload.update(overrides)
    return QuoteAcceptanceEvidence.model_validate(payload)


@pytest.mark.unit
@pytest.mark.gate
def test_fake_conforms_to_protocol() -> None:
    fake: ExternalWorkIntegration = DeterministicExternalWorkFake()
    assert isinstance(fake, ExternalWorkIntegration)


@pytest.mark.unit
@pytest.mark.gate
def test_adapter_creation_and_dependency_injection() -> None:
    fake = DeterministicExternalWorkFake()
    policy = _allow_policy()
    adapter = ExternalWorkAdapter(fake, side_effect_policy=policy)
    agent = ExternalContractorAdapterAgent(
        external_work=fake,
        side_effect_policy=policy,
    )
    assert adapter.integration is fake
    assert adapter.side_effect_policy is policy
    assert agent._external_work is fake
    assert agent._side_effect_policy is policy


@pytest.mark.unit
@pytest.mark.gate
def test_request_snapshot_quote_timeline_deliverables_evidence_mapping() -> None:
    fake = DeterministicExternalWorkFake()
    adapter = _adapter(fake)
    request = adapter.build_create_request(
        task_id="task-gec3",
        run_id="run-gec3",
        metadata=_meta(),
        message="ignored when scope present",
    )
    assert isinstance(request, ExternalWorkCreateRequest)
    assert request.task_id == "task-gec3"
    assert request.idempotency_key == "idem-gec3-1"
    result = adapter.create_and_map(request, principal_id="u1", tenant_id="tenant-a")
    assert result.used is True
    assert result.reason == "mapped"
    assert result.status == ExternalWorkStatus.QUOTE_AVAILABLE
    assert result.snapshot is not None
    assert result.snapshot.correlation.task_id == "task-gec3"
    assert result.snapshot.correlation.run_id == "run-gec3"
    assert result.snapshot.correlation.external_task_id.startswith("ext-gec3-")
    assert result.quote is not None
    assert result.quote.quote_id == "q-gec3-1"
    assert len(result.timeline) == 2
    assert len(result.deliverables) == 1
    assert result.deliverables[0].resource_uri.startswith("workspace://")
    assert len(result.evidence) == 1
    assert result.evidence[0].provider_id == "gec3_deterministic_fake"


@pytest.mark.unit
@pytest.mark.gate
def test_correlation_and_idempotency_preserved() -> None:
    fake = DeterministicExternalWorkFake()
    adapter = _adapter(fake)
    request = adapter.build_create_request(
        task_id="task-idem",
        run_id="run-idem",
        metadata=_meta(**{META_IDEMPOTENCY_KEY: "same-key"}),
    )
    first = adapter.create_and_map(
        request, enrich=False, principal_id="u1", tenant_id="tenant-a"
    )
    second = adapter.create_and_map(
        request, enrich=False, principal_id="u1", tenant_id="tenant-a"
    )
    assert fake.create_calls == 2  # invoked twice; provider returns same snapshot
    assert first.snapshot is not None and second.snapshot is not None
    assert first.snapshot.correlation.external_task_id == (
        second.snapshot.correlation.external_task_id
    )
    assert first.snapshot.correlation.idempotency_key == "same-key"
    assert first.snapshot.correlation.task_id == "task-idem"


@pytest.mark.unit
@pytest.mark.gate
def test_unsupported_capability_behavior() -> None:
    fake = DeterministicExternalWorkFake(
        capabilities=(ExternalWorkCapability.QUOTE_FIRST,),
    )
    adapter = _adapter(fake)
    request = adapter.build_create_request(
        task_id="task-cap",
        run_id="run-cap",
        metadata=_meta(**{META_IDEMPOTENCY_KEY: "idem-cap"}),
    )
    result = adapter.create_and_map(request, principal_id="u1", tenant_id="tenant-a")
    assert result.used is True
    assert ExternalWorkCapability.TIMELINE in result.unsupported_capabilities
    assert ExternalWorkCapability.DELIVERABLES in result.unsupported_capabilities
    assert ExternalWorkCapability.EVIDENCE_REFS in result.unsupported_capabilities
    assert result.timeline == ()
    assert result.deliverables == ()
    assert result.evidence == ()
    assert result.quote is not None


@pytest.mark.unit
@pytest.mark.gate
def test_structured_error_propagation() -> None:
    fake = DeterministicExternalWorkFake(unsupported_ops=frozenset({"create_work"}))
    adapter = _adapter(fake)
    request = adapter.build_create_request(
        task_id="task-err",
        run_id="run-err",
        metadata=_meta(**{META_IDEMPOTENCY_KEY: "idem-err"}),
    )
    result = adapter.create_and_map(request, principal_id="u1", tenant_id="tenant-a")
    assert result.used is False
    assert result.reason == "external_work_error"
    assert result.error_code == ExternalWorkErrorCode.OPERATION_NOT_SUPPORTED
    assert result.error_retryable is False


@pytest.mark.unit
@pytest.mark.gate
def test_forward_quote_acceptance_does_not_decide() -> None:
    fake = DeterministicExternalWorkFake()
    adapter = _adapter(fake)
    request = adapter.build_create_request(
        task_id="task-acc",
        run_id="run-acc",
        metadata=_meta(**{META_IDEMPOTENCY_KEY: "idem-acc"}),
    )
    created = adapter.create_and_map(
        request, enrich=False, principal_id="u1", tenant_id="tenant-a"
    )
    assert created.snapshot is not None and created.quote is not None
    acceptance = _acceptance(quote_id=created.quote.quote_id)
    forwarded = adapter.forward_quote_acceptance(
        created.snapshot.correlation,
        acceptance,
        idempotency_key="idem-accept-1",
    )
    assert forwarded.used is True
    assert forwarded.status == ExternalWorkStatus.ACCEPTED
    source = _ADAPTER_PY.read_text(encoding="utf-8")
    assert "submit_quote_acceptance" in source
    assert "Evidence ≠ authorization" in source or "does not decide" in source.lower()


@pytest.mark.unit
@pytest.mark.gate
def test_adapt_from_step_metadata_missing_integration() -> None:
    result = adapt_from_step_metadata(
        None,
        task_id="t1",
        run_id="r1",
        message="hi",
        metadata=_meta(),
    )
    assert result.used is False
    assert result.reason == "external_work_integration_missing"


@pytest.mark.unit
@pytest.mark.gate
def test_adapt_from_step_metadata_with_acceptance_forward() -> None:
    fake = DeterministicExternalWorkFake()
    policy = _allow_policy()
    # First create to learn quote id, then full path with acceptance in metadata.
    bootstrap = _adapter(fake, policy=policy).create_and_map(
        _adapter(fake, policy=policy).build_create_request(
            task_id="task-meta",
            run_id="run-meta",
            metadata=_meta(**{META_IDEMPOTENCY_KEY: "idem-meta-boot"}),
        ),
        enrich=False,
        principal_id="u1",
        tenant_id="tenant-a",
    )
    assert bootstrap.quote is not None
    result = adapt_from_step_metadata(
        fake,
        task_id="task-meta",
        run_id="run-meta",
        message="scope",
        metadata=_meta(
            **{
                META_IDEMPOTENCY_KEY: "idem-meta-boot",
                META_QUOTE_ACCEPTANCE: _acceptance(quote_id=bootstrap.quote.quote_id),
                META_ACCEPTANCE_IDEMPOTENCY_KEY: "idem-accept-meta",
            }
        ),
        side_effect_policy=policy,
    )
    assert result.used is True
    assert result.status == ExternalWorkStatus.ACCEPTED


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.gate
async def test_agent_run_with_injected_fake() -> None:
    fake = DeterministicExternalWorkFake()
    policy = _allow_policy()
    agent = ExternalContractorAdapterAgent(
        external_work=fake,
        side_effect_policy=policy,
    )
    result = await agent.run(
        AgentRunRequest(
            input="map external work",
            identity=RequestIdentity(tenant_id="t1", user_id="u1"),
            agent_id="external_contractor_adapter",
            metadata=_meta(
                **{
                    META_IDEMPOTENCY_KEY: "idem-agent-run",
                    "external_work.principal_id": "u1",
                    "external_work.tenant_id": "t1",
                }
            ),
        )
    )
    assert result.status == AgentRunStatus.SUCCEEDED
    output = result.output
    assert isinstance(output, dict)
    domain = output.get("domain_summary")
    assert isinstance(domain, dict)
    assert domain.get("used") is True
    # GEC-4: quote without continuation evidence surfaces a blocker (not a decision).
    assert domain.get("reason") == "continuation_blocked"
    assert isinstance(domain.get("continuation"), dict)
    assert domain["continuation"].get("reason") == "quote"
    assert "external_contractor.adapt" in str(output)


@pytest.mark.unit
@pytest.mark.gate
def test_no_transport_or_application_imports() -> None:
    forbidden_modules = {
        "httpx",
        "requests",
        "aiohttp",
        "urllib",
        "urllib3",
        "http.client",
        "applications",
        "governed_contractor_application",
    }
    for path in (_ADAPTER_PY, _AGENT_PY, _DOMAIN_PY):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".")[0]
                    assert root not in forbidden_modules, f"{path}: import {alias.name}"
            elif isinstance(node, ast.ImportFrom) and node.module:
                root = node.module.split(".")[0]
                assert root not in forbidden_modules, f"{path}: from {node.module}"


@pytest.mark.unit
@pytest.mark.gate
def test_no_provider_specific_branching() -> None:
    source = _ADAPTER_PY.read_text(encoding="utf-8")
    lowered = source.lower()
    for needle in (
        'provider == "a2a"',
        "provider == 'a2a'",
        'provider == "rest"',
        "provider == 'rest'",
        "provider.protocol",
        "if protocol ==",
    ):
        assert needle not in lowered


@pytest.mark.unit
@pytest.mark.gate
def test_external_work_error_is_structured_not_http() -> None:
    err = ExternalWorkError(
        "boom",
        code=ExternalWorkErrorCode.TRANSIENT_REMOTE_FAILURE,
        provider_id="gec3_deterministic_fake",
    )
    assert err.retryable is True
    assert not isinstance(err, OSError)
