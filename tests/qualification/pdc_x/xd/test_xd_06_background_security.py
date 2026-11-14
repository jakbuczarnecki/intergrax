# © Artur Czarnecki. All rights reserved.

"""XD-06 (Background <-> Security): retried tasks re-authorize; revocation mid-retry fails closed.

PDC-D6-R009 (credential revoked mid-retry denied) and PDC-D6-R019 (cross-tenant negative).

The task runs through the real queue (``DocumentStoreTaskQueue``) and the real retrying worker
(``DocumentStoreTaskWorker`` with a ``RetryPolicy``); the handler obtains its credential through the
real ``ScopedCredentialBroker`` on every attempt. Fakes are limited to the external credential
vendor (the secrets store) and to the vendor API behind the handler.

Test-side only: the ``CredentialScopeAdmissionPort`` implementation. The product defines the port
but ships no implementation and no composition root that wires the broker into the worker path
(reported as a finding).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

import pytest
from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.integrations.contracts.credential import (
    CredentialRef,
    CredentialScopeAdmissionDecision,
    CredentialScopeAdmissionDeniedError,
    CredentialScopeMismatchError,
    CredentialUseGrant,
    CredentialUseScope,
)
from intergrax.integrations.credentials.broker import ScopedCredentialBroker
from intergrax.integrations.credentials.secrets_store_resolver import SecretsStoreCredentialResolver
from intergrax.queueing.contracts.task_queue import TaskHandle, TaskRequest, TaskStatus
from intergrax.queueing.providers.document_store import DocumentStoreTaskQueue
from intergrax.queueing.providers.document_store.colocated_worker import DocumentStoreTaskWorker
from intergrax.queueing.worker.execution import RetryableHandlerError
from intergrax.queueing.worker.registry import TaskExecutionRegistry
from intergrax.queueing.worker.retry_policy import RetryPolicy
from intergrax.runtime.background_execution.admission_wiring import (
    wire_background_execution_admission_dependencies,
)
from intergrax.runtime.observability.document_store_causal_evidence_persistence import (
    wire_causal_evidence_persistence,
)
from intergrax.runtime.sandbox.network_egress import canonicalize_network_egress_allowlist
from intergrax.tools.execution_models import ToolExecutionResult
from pydantic import BaseModel
from tests.qualification.pdc_x.d6._support import CountingSecretsStore, generated_secret

pytestmark = [pytest.mark.integration]

TASK = "pdc.xd06.vendor_call"
TENANT_A = "tenant-a"
TENANT_B = "tenant-b"
SECRET_PATH_A = "secrets/tenant-a/vendor-api"
SECRET_PATH_B = "secrets/tenant-b/vendor-api"
_HOSTS = canonicalize_network_egress_allowlist(("https://api.vendor.com:443",))
_EXECUTION_ID = "exec-xd06"


class _Out(BaseModel):
    ok: bool = True


def _grant(tenant_id: str, secret_path: str, grant_id: str) -> CredentialUseGrant:
    return CredentialUseGrant(
        grant_id=grant_id,
        credential_ref=CredentialRef.from_secret_path(
            provider_id="vendor_api", secret_path=secret_path, tenant_id=tenant_id
        ),
        tenant_id=tenant_id,
        provider_id="vendor_api",
        integration_id="vendor_api:issue_tracker",
        operation="vendor.orders.write",
        execution_id=_EXECUTION_ID,
        target_scope=_HOSTS,
        expires_at=datetime.now(UTC) + timedelta(minutes=10),
    )


def _scope(tenant_id: str) -> CredentialUseScope:
    return CredentialUseScope(
        tenant_id=tenant_id,
        provider_id="vendor_api",
        integration_id="vendor_api:issue_tracker",
        operation="vendor.orders.write",
        execution_id=_EXECUTION_ID,
        target_scope=_HOSTS,
    )


class _RevocableAdmission:
    """Test-side ``CredentialScopeAdmissionPort``: a revocation list consulted on every call."""

    def __init__(self) -> None:
        self.revoked: set[str] = set()
        self.calls: list[str] = []

    def admit(self, grant: CredentialUseGrant, scope: CredentialUseScope) -> CredentialScopeAdmissionDecision:
        self.calls.append(grant.grant_id)
        if grant.grant_id in self.revoked:
            return CredentialScopeAdmissionDecision.DENY
        return CredentialScopeAdmissionDecision.ALLOW


@dataclass
class _Vendor:
    """External vendor API fake: records committed effects; can fail transiently before the effect."""

    fail_first_n: int = 0
    attempts: int = 0
    effects: list[tuple[str, str]] = field(default_factory=list)
    credentials_seen: list[str] = field(default_factory=list)

    def call(self, tenant_id: str, order: str, credential: str) -> None:
        self.attempts += 1
        self.credentials_seen.append(credential)
        if self.attempts <= self.fail_first_n:
            raise RetryableHandlerError("vendor 503 before commit")
        if (tenant_id, order) not in self.effects:  # the vendor dedupes on its own order id
            self.effects.append((tenant_id, order))


class _Harness:
    def __init__(self, *, vendor: _Vendor, max_retries: int = 3) -> None:
        self.secrets = CountingSecretsStore({SECRET_PATH_A: generated_secret(), SECRET_PATH_B: generated_secret()})
        self.admission = _RevocableAdmission()
        self.vendor = vendor
        self.grants = {
            "grant-a": _grant(TENANT_A, SECRET_PATH_A, "grant-a"),
            "grant-b": _grant(TENANT_B, SECRET_PATH_B, "grant-b"),
        }
        self.broker = ScopedCredentialBroker(
            resolver=SecretsStoreCredentialResolver(self.secrets),
            admission=self.admission,
        )
        self.store = InMemoryDocumentStore()
        self.queue = DocumentStoreTaskQueue(self.store)
        registry = TaskExecutionRegistry()
        registry.register(TASK, self._handler)
        admission_deps = wire_background_execution_admission_dependencies(document_store=self.store)
        self.worker = DocumentStoreTaskWorker(
            self.queue,
            registry,
            identity_persistence=admission_deps.identity_persistence,
            attempt_lifecycle=admission_deps.attempt_lifecycle,
            execution_terminal=admission_deps.execution_terminal,
            causal_evidence_persistence=wire_causal_evidence_persistence(document_store=self.store),
            retry_policy=RetryPolicy(
                max_retries=max_retries,
                initial_backoff_seconds=0.0,
                backoff_multiplier=1.0,
                max_backoff_seconds=None,
                jitter=False,
                retry_on_lock_conflict=True,
                retry_on_handler_exception=True,
            ),
        )
        self.on_attempt: dict[int, object] = {}

    def _handler(self, *, tenant_id, run_id, payload, idempotency_key, execution_identity):  # type: ignore[no-untyped-def]
        _ = run_id, idempotency_key, execution_identity
        request = json.loads(payload.decode("utf-8"))
        hook = self.on_attempt.get(len(self.admission.calls) + 1)
        if callable(hook):
            hook()
        # Every attempt re-authorizes: grant validation + admission + secret read happen per call.
        resolved = self.broker.resolve_scoped(self.grants[request["grant"]], _scope(tenant_id))
        self.vendor.call(tenant_id, request["order"], resolved.resolved_credential.value)
        return ToolExecutionResult.ok(_Out())

    def enqueue(self, tenant_id: str, grant: str, order: str) -> TaskHandle:
        return self.queue.enqueue(
            TaskRequest(
                tenant_id=tenant_id,
                run_id=f"run-{tenant_id}-{order}",
                task_name=TASK,
                payload=json.dumps({"grant": grant, "order": order}).encode("utf-8"),
            )
        )

    def drain(self, *, bound: int = 12) -> int:
        total = 0
        for _ in range(bound):
            claimed = self.worker.drain_once()
            if claimed == 0:
                return total
            total += claimed
        raise AssertionError("worker did not become idle within the drain bound")


def test_xd06_every_retry_reauthorizes_and_the_side_effect_lands_once() -> None:
    harness = _Harness(vendor=_Vendor(fail_first_n=2))
    handle = harness.enqueue(TENANT_A, "grant-a", "order-1")

    assert harness.drain() == 3  # two transient failures, then success
    assert harness.queue.get_status(handle) is TaskStatus.SUCCEEDED
    result = harness.queue.get_result(handle)
    assert result is not None and result.attempts == 3
    assert harness.admission.calls == ["grant-a"] * 3, "admission is re-evaluated on every attempt"
    assert len(harness.secrets.reads) == 3
    assert harness.vendor.effects == [(TENANT_A, "order-1")], "exactly one committed side effect"


def test_xd06_credential_revoked_mid_retry_fails_closed_without_side_effect() -> None:
    harness = _Harness(vendor=_Vendor(fail_first_n=1))
    # Revoke between attempt 1 (transient failure) and attempt 2 (the retry).
    harness.on_attempt[2] = lambda: harness.admission.revoked.add("grant-a")
    handle = harness.enqueue(TENANT_A, "grant-a", "order-2")

    assert harness.drain() == 2  # attempt 1 failed transiently, the retry was denied and not retried again
    assert harness.queue.get_status(handle) is TaskStatus.FAILED
    result = harness.queue.get_result(handle)
    assert result is not None and result.dead_letter is None
    assert "admission denied" in (result.error_message or "")
    assert harness.admission.calls == ["grant-a", "grant-a"]
    assert len(harness.secrets.reads) == 1, "the revoked retry never reads the secret store"
    assert harness.vendor.effects == [], "no side effect after revocation"
    assert harness.vendor.attempts == 1
    with pytest.raises(CredentialScopeAdmissionDeniedError):
        harness.broker.resolve_scoped(harness.grants["grant-a"], _scope(TENANT_A))


def test_xd06_secret_removed_at_the_vendor_between_attempts_also_fails_closed() -> None:
    harness = _Harness(vendor=_Vendor(fail_first_n=1))
    harness.on_attempt[2] = lambda: harness.secrets.delete_secret(SECRET_PATH_A)
    handle = harness.enqueue(TENANT_A, "grant-a", "order-3")

    assert harness.drain() == 2
    assert harness.queue.get_status(handle) is TaskStatus.FAILED
    assert harness.vendor.effects == []


def test_xd06_a_task_of_another_tenant_cannot_use_a_foreign_grant() -> None:
    harness = _Harness(vendor=_Vendor())
    foreign = harness.enqueue(TENANT_B, "grant-a", "order-x")  # tenant B names tenant A's grant
    own = harness.enqueue(TENANT_B, "grant-b", "order-y")

    harness.drain()
    assert harness.queue.get_status(foreign) is TaskStatus.FAILED
    failed = harness.queue.get_result(foreign)
    assert failed is not None and "tenant" in (failed.error_message or "").lower()
    assert SECRET_PATH_A not in harness.secrets.reads, "tenant A's secret path was never touched"
    assert (TENANT_B, "order-x") not in harness.vendor.effects
    # Tenant B's own grant keeps working, and only against tenant B's secret.
    assert harness.queue.get_status(own) is TaskStatus.SUCCEEDED
    assert harness.vendor.effects == [(TENANT_B, "order-y")]
    assert harness.secrets.reads == [SECRET_PATH_B]
    with pytest.raises(CredentialScopeMismatchError):
        harness.broker.resolve_scoped(harness.grants["grant-a"], _scope(TENANT_B))


def test_xd06_no_raw_secret_in_task_state_results_or_failures() -> None:
    harness = _Harness(vendor=_Vendor(fail_first_n=1))
    harness.on_attempt[2] = lambda: harness.admission.revoked.add("grant-a")
    failed = harness.enqueue(TENANT_A, "grant-a", "order-4")
    ok = harness.enqueue(TENANT_B, "grant-b", "order-5")
    harness.drain()
    secret_values = list(harness.secrets.values.values())
    surfaces = repr((harness.queue.get_result(failed), harness.queue.get_result(ok)))
    surfaces += repr(harness.store._rows)  # every durable row the queue and worker wrote
    assert len(harness.store._rows) > 0
    for value in secret_values:
        assert value not in surfaces
