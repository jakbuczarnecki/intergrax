# © Artur Czarnecki. All rights reserved.

"""GR-7-A8-R3 — explicit reliability-aware dispatch composition (no hidden cast)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pytest

from applications.governed_contractor_application.host.provider_invocation_lifecycle import (
    GovernedProviderInvocationDispatchGate,
)
from decimal import Decimal

from external_contractor_adapter.external_work_adapter import (
    META_IDEMPOTENCY_KEY,
    META_PROVIDER_ID,
    META_SCOPE_DESCRIPTION,
    META_SCOPE_DIGEST,
    ExternalWorkAdapter,
)
from intergrax.contracts.money import MoneyAmount
from external_contractor_adapter.tests.fakes.adapter_test_wiring import (
    EXTERNAL_WORK_TEST_RUN_ID,
    EXTERNAL_WORK_TEST_TASK_ID,
    allow_adapter,
    bound_external_work_test_execution,
    default_workspace_meta,
)
from external_contractor_adapter.tests.fakes.deterministic_external_work import (
    DeterministicExternalWorkFake,
)
from intergrax.contracts.enterprise_reliability.provider_invocation_reliability_evidence import (
    ProviderInvocationReliabilityFact,
    ProviderInvocationReliabilityTracePhase,
)
from intergrax.contracts.external_work_provider_capabilities import (
    quote_first_partner_capability_fixture,
)
from intergrax.contracts.provider_invocation import ProviderInvocation
from intergrax.runtime.enterprise_reliability.provider_invocation_reliability_dispatch_context import (
    ProviderInvocationReliabilityDispatchContext,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_ADAPTER_PY = Path(__file__).resolve().parents[4] / "agents" / "external_contractor_adapter" / "external_work_adapter.py"
_T0 = datetime(2026, 9, 17, 14, 0, 0, tzinfo=timezone.utc)
_PROVIDER = "partner-demo"
_TENANT = "tenant-a"
_PRINCIPAL = "u1"
_DIGEST = "sha256:" + ("ab" * 32)


def _meta() -> dict[str, object]:
    return {
        META_PROVIDER_ID: _PROVIDER,
        META_SCOPE_DESCRIPTION: "r3 scope",
        META_SCOPE_DIGEST: _DIGEST,
        META_IDEMPOTENCY_KEY: "idem-gr7-a8-r3",
        **default_workspace_meta(),
        "external_work.budget_limit": MoneyAmount(amount=Decimal("40.00"), currency="USD"),
        "external_work.principal_id": _PRINCIPAL,
        "external_work.tenant_id": _TENANT,
    }


def _invocation() -> ProviderInvocation:
    return ProviderInvocation(
        invocation_id="inv-r3",
        provider_id=_PROVIDER,
        operation="create_work",
        task_id=EXTERNAL_WORK_TEST_TASK_ID,
        run_id=EXTERNAL_WORK_TEST_RUN_ID,
        request_digest="sha256:" + ("cd" * 32),
        started_at=_T0,
    )


@dataclass
class _BaseOnlyDispatchPort:
    dispatch_calls: int = 0
    execute_calls: int = 0

    def dispatch_after_intent_persisted(self, invocation: ProviderInvocation, execute):
        self.dispatch_calls += 1
        self.execute_calls += 1
        return execute()


@dataclass
class _AwareDispatchPort:
    dispatch_calls: int = 0
    reliability_contexts: list[ProviderInvocationReliabilityDispatchContext | None] = field(
        default_factory=list,
    )
    execute_calls: int = 0

    def dispatch_after_intent_persisted(
        self,
        invocation: ProviderInvocation,
        execute,
        *,
        reliability_dispatch: ProviderInvocationReliabilityDispatchContext | None = None,
    ):
        self.dispatch_calls += 1
        self.reliability_contexts.append(reliability_dispatch)
        self.execute_calls += 1
        return execute()


@dataclass
class _RecordingObserver:
    facts: list[ProviderInvocationReliabilityFact] = field(default_factory=list)

    def observe_provider_invocation_reliability_fact(
        self,
        fact: ProviderInvocationReliabilityFact,
    ) -> None:
        self.facts.append(fact)


def test_custom_base_port_observer_off_works() -> None:
    fake = DeterministicExternalWorkFake()
    dispatch = _BaseOnlyDispatchPort()
    adapter, _ = allow_adapter(fake, invocation_dispatch=dispatch)
    with bound_external_work_test_execution():
        result = adapter.create_and_map(
            adapter.build_create_request(
                task_id=EXTERNAL_WORK_TEST_TASK_ID,
                run_id=EXTERNAL_WORK_TEST_RUN_ID,
                metadata=_meta(),
            ),
            principal_id=_PRINCIPAL,
            tenant_id=_TENANT,
            provider_invocation=_invocation(),
        )
    assert result.used is True
    assert dispatch.dispatch_calls == 1
    assert dispatch.execute_calls == 1


def test_custom_base_port_observer_on_no_type_error_provider_unchanged() -> None:
    fake = DeterministicExternalWorkFake()
    dispatch = _BaseOnlyDispatchPort()
    observer = _RecordingObserver()
    adapter, _ = allow_adapter(fake, invocation_dispatch=dispatch)
    adapter = ExternalWorkAdapter(
        fake,
        authorization_boundary=adapter.authorization_boundary,
        invocation_dispatch=dispatch,
        reliability_evidence_observer=observer,
        provider_capabilities=quote_first_partner_capability_fixture(provider_id=_PROVIDER),
        clock=lambda: _T0,
    )
    with bound_external_work_test_execution():
        result = adapter.create_and_map(
            adapter.build_create_request(
                task_id=EXTERNAL_WORK_TEST_TASK_ID,
                run_id=EXTERNAL_WORK_TEST_RUN_ID,
                metadata=_meta(),
            ),
            principal_id=_PRINCIPAL,
            tenant_id=_TENANT,
            provider_invocation=_invocation(),
        )
    assert result.used is True
    assert dispatch.dispatch_calls == 1
    assert dispatch.execute_calls == 1


def test_custom_aware_port_observer_on_receives_reliability_context() -> None:
    fake = DeterministicExternalWorkFake()
    base = _BaseOnlyDispatchPort()
    aware = _AwareDispatchPort()
    observer = _RecordingObserver()
    adapter, _ = allow_adapter(fake, invocation_dispatch=base)
    adapter = ExternalWorkAdapter(
        fake,
        authorization_boundary=adapter.authorization_boundary,
        invocation_dispatch=base,
        reliability_aware_invocation_dispatch=aware,
        reliability_evidence_observer=observer,
        provider_capabilities=quote_first_partner_capability_fixture(provider_id=_PROVIDER),
        clock=lambda: _T0,
    )
    with bound_external_work_test_execution():
        result = adapter.create_and_map(
            adapter.build_create_request(
                task_id=EXTERNAL_WORK_TEST_TASK_ID,
                run_id=EXTERNAL_WORK_TEST_RUN_ID,
                metadata=_meta(),
            ),
            principal_id=_PRINCIPAL,
            tenant_id=_TENANT,
            provider_invocation=_invocation(),
        )
    assert result.used is True
    assert base.dispatch_calls == 0
    assert aware.dispatch_calls == 1
    assert aware.reliability_contexts[0] is not None
    assert aware.reliability_contexts[0].observer is observer


def test_same_gate_as_base_and_aware_emits_early_lifecycle_with_observer() -> None:
    from applications.governed_contractor_application.tests.host.durable_provider_invocation_test_store import (
        DurableTestProviderInvocationStore,
    )

    fake = DeterministicExternalWorkFake()
    store = DurableTestProviderInvocationStore()
    gate = GovernedProviderInvocationDispatchGate(store=store, clock=lambda: _T0)
    observer = _RecordingObserver()
    adapter, _ = allow_adapter(fake, invocation_dispatch=gate)
    adapter = ExternalWorkAdapter(
        fake,
        authorization_boundary=adapter.authorization_boundary,
        invocation_dispatch=gate,
        reliability_aware_invocation_dispatch=gate,
        reliability_evidence_observer=observer,
        provider_capabilities=quote_first_partner_capability_fixture(provider_id=_PROVIDER),
        clock=lambda: _T0,
    )
    with bound_external_work_test_execution():
        result = adapter.create_and_map(
            adapter.build_create_request(
                task_id=EXTERNAL_WORK_TEST_TASK_ID,
                run_id=EXTERNAL_WORK_TEST_RUN_ID,
                metadata=_meta(),
            ),
            principal_id=_PRINCIPAL,
            tenant_id=_TENANT,
            provider_invocation=_invocation(),
        )
    assert result.used is True
    phases = {f.phase for f in observer.facts}
    assert ProviderInvocationReliabilityTracePhase.GOVERNANCE_AUTHORIZED in phases
    assert ProviderInvocationReliabilityTracePhase.INTENT_PERSISTED in phases
    assert ProviderInvocationReliabilityTracePhase.DISPATCH_ATTEMPTED in phases


def test_aware_port_observer_off_uses_base_path() -> None:
    fake = DeterministicExternalWorkFake()
    base = _BaseOnlyDispatchPort()
    aware = _AwareDispatchPort()
    adapter, _ = allow_adapter(fake, invocation_dispatch=base)
    adapter = ExternalWorkAdapter(
        fake,
        authorization_boundary=adapter.authorization_boundary,
        invocation_dispatch=base,
        reliability_aware_invocation_dispatch=aware,
    )
    with bound_external_work_test_execution():
        result = adapter.create_and_map(
            adapter.build_create_request(
                task_id=EXTERNAL_WORK_TEST_TASK_ID,
                run_id=EXTERNAL_WORK_TEST_RUN_ID,
                metadata=_meta(),
            ),
            principal_id=_PRINCIPAL,
            tenant_id=_TENANT,
            provider_invocation=_invocation(),
        )
    assert result.used is True
    assert base.dispatch_calls == 1
    assert aware.dispatch_calls == 0


@pytest.mark.gate
def test_adapter_dispatch_path_has_no_reliability_aware_cast() -> None:
    source = _ADAPTER_PY.read_text(encoding="utf-8")
    assert "cast(" not in source
    assert "ProviderInvocationReliabilityAwareDispatchPort" in source
    tracked = source.split("def _tracked_execute", maxsplit=1)[1].split("def _capture_authorization", maxsplit=1)[0]
    assert "hasattr(" not in tracked
    assert "getattr(" not in tracked
    assert re.search(r"except\s+TypeError", tracked) is None


@pytest.mark.gate
def test_production_composition_declares_aware_dispatch_binding() -> None:
    path = (
        Path(__file__).resolve().parents[4]
        / "applications"
        / "governed_contractor_application"
        / "host"
        / "production_external_work_composition.py"
    )
    source = path.read_text(encoding="utf-8")
    assert "reliability_aware_invocation_dispatch=invocation_dispatch" in source
