# © Artur Czarnecki. All rights reserved.

"""XD-10 (Artifacts <-> Security): cross-tenant read denied, TTL deletion honored, no raw secret in artifacts.

PDC-D6-R019 (cross-tenant negatives), PDC-D6-R022 (TTL deletion honored) and PDC-D8-R005
(tenant/workspace scope of artifacts).

Integrated version of ``d8/test_xd10_tenant_and_ttl.py``. Reads go through the governed port over the
real governance pipeline on the durable SQLite artifact store; TTL deletion runs as real
``WorkerRuntime`` background work per tenant; secrets are resolved by the real
``ScopedCredentialBroker`` and are then searched for in every artifact surface (store bytes, metadata,
audit evidence). Fakes: the credential vendor (secrets store) and the object storage provider only.
"""

from __future__ import annotations

import json
import random
import string
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from intergrax.background_tasks.definition import TaskDefinition
from intergrax.contracts.agent_runtime_governance import ToolAuthorizationDecisionState
from intergrax.contracts.artifact_ref import ArtifactClass, ArtifactRetentionClass
from intergrax.integrations.contracts.credential import (
    CredentialRef,
    CredentialScopeAdmissionDecision,
    CredentialUseGrant,
    CredentialUseScope,
)
from intergrax.integrations.credentials.broker import ScopedCredentialBroker
from intergrax.integrations.credentials.secrets_store_resolver import SecretsStoreCredentialResolver
from intergrax.queueing.contracts.task_queue import TaskRequest, TaskStatus
from intergrax.runtime.nexus.artifacts.context_reference import ArtifactReferenceResolver
from intergrax.runtime.nexus.artifacts.governed_access import (
    ArtifactAccessDenied,
    ArtifactNotFound,
    ArtifactReadRequest,
)
from intergrax.runtime.nexus.artifacts.retention import (
    ARTIFACT_RETENTION_SWEEP_TASK_NAME,
    ArtifactRetentionSchedule,
    ArtifactRetentionSweeper,
    make_retention_sweep_handler,
)
from intergrax.runtime.nexus.artifacts.spill import ToolOutputSpiller
from intergrax.runtime.sandbox.network_egress import canonicalize_network_egress_allowlist
from tests.qualification.pdc_x.d6._support import CountingSecretsStore, generated_secret
from tests.qualification.pdc_x.d8._d8_support import (
    T0,
    TENANT_A,
    TENANT_B,
    FakeObjectStorage,
    MutableClock,
    make_artifact,
    make_context,
    make_governance,
    make_port,
    reader_grants,
    sqlite_store,
)
from tests.qualification.pdc_x.xd._support import build_sqlite_worker

pytestmark = [pytest.mark.integration]

BIG_PREFIX = "row,value,notes\n"


def _req(tenant_id: str, artifact_id: str = "art-1", *, cls: ArtifactClass = ArtifactClass.EXECUTION_EVIDENCE):
    return ArtifactReadRequest(artifact_class=cls, tenant_id=tenant_id, artifact_id=artifact_id)


def _two_tenant_stack(tmp_path: Path):  # type: ignore[no-untyped-def]
    store = sqlite_store(tmp_path)
    store.put(make_artifact(tenant_id=TENANT_A, data=b"tenant-a secret report"))
    store.put(make_artifact(tenant_id=TENANT_B, data=b"tenant-b secret report"))
    harness = make_governance(granted={**reader_grants(tenant_id=TENANT_A), **reader_grants(tenant_id=TENANT_B)})
    port, _ = make_port(store, harness=harness)
    return store, port, harness


# ----------------------------------------------------------------------- cross-tenant negatives (D6-R019, D8-R005)


def test_xd10_cross_tenant_read_is_denied_audited_and_never_reaches_the_store(tmp_path: Path) -> None:
    store, port, harness = _two_tenant_stack(tmp_path)
    reads: list[tuple[str, str]] = []
    original = store.get

    def spy(tenant_id, artifact_id, *args, **kwargs):  # type: ignore[no-untyped-def]
        reads.append((tenant_id, artifact_id))
        return original(tenant_id, artifact_id, *args, **kwargs)

    store.get = spy  # type: ignore[method-assign]
    reader_a, reader_b = make_context(tenant_id=TENANT_A), make_context(tenant_id=TENANT_B)

    assert port.read(reader_a, _req(TENANT_A)).content == b"tenant-a secret report"
    assert port.read(reader_b, _req(TENANT_B)).content == b"tenant-b secret report"
    reads.clear()

    for reader, foreign in ((reader_a, TENANT_B), (reader_b, TENANT_A)):
        with pytest.raises(ArtifactAccessDenied) as exc:
            port.read(reader, _req(foreign))
        assert exc.value.reason_code == "tenant_mismatch"
    assert reads == [], "a cross-tenant request is refused before any store access"
    denied = [
        e
        for e in harness.sink.events
        if e.capability == "artifact.read.outcome" and e.decision is ToolAuthorizationDecisionState.DENY
    ]
    assert {e.tenant_id for e in denied} == {TENANT_A, TENANT_B}  # each denial is recorded under the reader


def test_xd10_cross_tenant_reference_token_cannot_be_resolved(tmp_path: Path) -> None:
    store = sqlite_store(tmp_path)
    clock = MutableClock()
    spiller = ToolOutputSpiller(store, threshold_bytes=64, schedule=ArtifactRetentionSchedule.default(), clock=clock)
    body = (BIG_PREFIX + "alpha,1\n" * 50).encode()
    token = spiller.spill_if_large(
        tenant_id=TENANT_A, run_id="run-1", tool_id="t", agent_id="agent-1", content=body
    ).reference
    assert token is not None

    harness = make_governance(granted={**reader_grants(tenant_id=TENANT_A), **reader_grants(tenant_id=TENANT_B)})
    port, _ = make_port(store, harness=harness, clock=clock)
    resolver = ArtifactReferenceResolver(port)
    assert resolver.resolve(make_context(tenant_id=TENANT_A), token) == body
    with pytest.raises(ArtifactAccessDenied) as exc:
        resolver.resolve(make_context(tenant_id=TENANT_B), token)  # tenant B holds tenant A's marker
    assert exc.value.reason_code == "tenant_mismatch"
    # Re-pointing the marker at tenant B's namespace finds nothing: ids are tenant-scoped.
    forged = token.model_copy(update={"tenant_id": TENANT_B})
    with pytest.raises(ArtifactNotFound):
        resolver.resolve(make_context(tenant_id=TENANT_B), forged)


def test_xd10_cross_tenant_download_is_denied_and_no_url_is_issued(tmp_path: Path) -> None:
    store, _, harness = _two_tenant_stack(tmp_path)
    storage = FakeObjectStorage()
    port, _ = make_port(store, harness=harness, object_storage=storage)
    with pytest.raises(ArtifactAccessDenied):
        port.issue_download_url(make_context(tenant_id=TENANT_A), _req(TENANT_B), ttl_seconds=60)
    assert storage.presign_calls == [], "no presigned URL is signed for a foreign tenant"


def test_xd10_agent_without_a_grant_is_denied_even_in_its_own_tenant(tmp_path: Path) -> None:
    store = sqlite_store(tmp_path)
    store.put(make_artifact(tenant_id=TENANT_A))
    harness = make_governance(granted={})  # no capability granted to anyone
    port, _ = make_port(store, harness=harness)
    with pytest.raises(ArtifactAccessDenied):
        port.read(make_context(tenant_id=TENANT_A), _req(TENANT_A))
    assert harness.sink.events


# ----------------------------------------------------------------------- TTL through real background work (D6-R022)


def _sweep_worker(tmp_path: Path, store, clock, harness):  # type: ignore[no-untyped-def]
    sweeper = ArtifactRetentionSweeper(store, clock=clock, audit=harness.recorder)
    return build_sqlite_worker(
        tmp_path,
        [
            TaskDefinition(
                task_name=ARTIFACT_RETENTION_SWEEP_TASK_NAME,
                payload_schema=dict,
                handler=make_retention_sweep_handler(sweeper),
            )
        ],
    )


def _run_sweep(worker, tenant_id: str, run_id: str):  # type: ignore[no-untyped-def]
    request = TaskRequest(
        tenant_id=tenant_id,
        run_id=run_id,
        task_name=ARTIFACT_RETENTION_SWEEP_TASK_NAME,
        payload=json.dumps({"limit": 100}).encode("utf-8"),
    )
    return worker.process_request(request, task_id=run_id)


def test_xd10_ttl_sweep_runs_per_tenant_as_worker_runtime_work_and_never_crosses_tenants(tmp_path: Path) -> None:
    store = sqlite_store(tmp_path)
    clock = MutableClock()
    for tenant in (TENANT_A, TENANT_B):
        store.put(
            make_artifact(
                tenant_id=tenant,
                artifact_id="ephemeral",
                retention_class=ArtifactRetentionClass.EPHEMERAL,
                expires_at_utc=T0 + timedelta(hours=1),
            )
        )
        store.put(
            make_artifact(
                tenant_id=tenant,
                artifact_id="evidence",
                retention_class=ArtifactRetentionClass.AUDIT_EVIDENCE,
                expires_at_utc=T0 + timedelta(hours=1),
            )
        )
    harness = make_governance()
    worker = _sweep_worker(tmp_path, store, clock, harness)

    clock.advance(hours=2)
    assert _run_sweep(worker, TENANT_A, "sweep-a").status is TaskStatus.SUCCEEDED
    with pytest.raises(KeyError):
        store.get(TENANT_A, "ephemeral")
    assert store.get(TENANT_A, "evidence")
    # Tenant B has not been swept yet: its expired artifact is untouched by tenant A's run.
    assert store.get(TENANT_B, "ephemeral")

    assert _run_sweep(worker, TENANT_B, "sweep-b").status is TaskStatus.SUCCEEDED
    with pytest.raises(KeyError):
        store.get(TENANT_B, "ephemeral")
    assert store.get(TENANT_B, "evidence")

    deletes = [e for e in harness.sink.events if e.capability == "artifact.delete.outcome"]
    allowed = {e.tenant_id for e in deletes if e.decision is ToolAuthorizationDecisionState.ALLOW}
    assert allowed == {TENANT_A, TENANT_B}
    assert all(e.agent_id == "artifact_retention_sweep" for e in deletes)

    # TTL state is durable: after a restart the deleted artifact stays gone, evidence stays.
    del store
    reopened = sqlite_store(tmp_path)
    with pytest.raises(KeyError):
        reopened.get(TENANT_A, "ephemeral")
    assert reopened.get(TENANT_A, "evidence") and reopened.get(TENANT_B, "evidence")


def test_xd10_nothing_is_deleted_before_the_ttl_elapses(tmp_path: Path) -> None:
    store = sqlite_store(tmp_path)
    clock = MutableClock()
    store.put(
        make_artifact(
            artifact_id="ephemeral",
            retention_class=ArtifactRetentionClass.EPHEMERAL,
            expires_at_utc=T0 + timedelta(hours=1),
        )
    )
    worker = _sweep_worker(tmp_path, store, clock, make_governance())
    clock.advance(minutes=59)
    assert _run_sweep(worker, TENANT_A, "sweep-early").status is TaskStatus.SUCCEEDED
    assert store.get(TENANT_A, "ephemeral")


# ----------------------------------------------------------------------- no raw secret in artifacts


class _AllowAdmission:
    def admit(self, grant: CredentialUseGrant, scope: CredentialUseScope) -> CredentialScopeAdmissionDecision:
        return CredentialScopeAdmissionDecision.ALLOW


def _credentials(secret: str):  # type: ignore[no-untyped-def]
    hosts = canonicalize_network_egress_allowlist(("https://api.vendor.com:443",))
    path = "secrets/tenant-a/vendor-api"
    grant = CredentialUseGrant(
        grant_id="grant-1",
        credential_ref=CredentialRef.from_secret_path(provider_id="vendor_api", secret_path=path, tenant_id=TENANT_A),
        tenant_id=TENANT_A,
        provider_id="vendor_api",
        integration_id="vendor_api:issue_tracker",
        operation="vendor.orders.read",
        execution_id="exec-xd10",
        target_scope=hosts,
        expires_at=datetime.now(UTC) + timedelta(minutes=10),
    )
    scope = CredentialUseScope(
        tenant_id=TENANT_A,
        provider_id="vendor_api",
        integration_id="vendor_api:issue_tracker",
        operation="vendor.orders.read",
        execution_id="exec-xd10",
        target_scope=hosts,
    )
    broker = ScopedCredentialBroker(
        resolver=SecretsStoreCredentialResolver(CountingSecretsStore({path: secret})),
        admission=_AllowAdmission(),
    )
    return broker, grant, scope


def test_xd10_credential_used_by_a_tool_never_lands_in_artifacts_or_evidence(tmp_path: Path) -> None:
    secret = generated_secret()
    broker, grant, scope = _credentials(secret)
    store = sqlite_store(tmp_path, "artifacts.sqlite3")
    clock = MutableClock()
    harness = make_governance(granted=reader_grants())
    port, _ = make_port(store, harness=harness, clock=clock)

    # The "tool": resolves the credential at the use site, calls the vendor, spills its large output.
    resolved = broker.resolve_scoped(grant, scope)
    assert resolved.resolved_credential.value == secret
    vendor_rows = BIG_PREFIX + "order,1,shipped\n" * 2000  # vendor response; does not echo the credential
    spill = ToolOutputSpiller(store, threshold_bytes=4096, clock=clock).spill_if_large(
        tenant_id=TENANT_A, run_id="run-1", tool_id="vendor.orders.read", agent_id="agent-1", content=vendor_rows
    )
    assert spill.reference is not None
    # Evidence of the use is stored as a durable artifact: the broker's safe evidence only.
    store.put(
        make_artifact(
            artifact_id="credential-use",
            data=repr(resolved.use_evidence).encode("utf-8"),
            retention_class=ArtifactRetentionClass.AUDIT_EVIDENCE,
        )
    )
    assert ArtifactReferenceResolver(port).resolve(make_context(), spill.reference) == vendor_rows.encode()

    surfaces = [
        (tmp_path / "artifacts.sqlite3").read_bytes(),
        repr([store.get(TENANT_A, a.artifact_id) for a in store.list_for_run(TENANT_A, "run-1")]).encode(),
        repr(harness.sink.events).encode(),
        repr(spill).encode(),
    ]
    for path in tmp_path.glob("artifacts.sqlite3*"):
        surfaces.append(path.read_bytes())  # WAL / journal files too
    for surface in surfaces:
        assert secret.encode() not in surface


def test_xd10_tool_output_echoing_a_credential_is_not_persisted_raw(tmp_path: Path) -> None:
    alphabet = string.ascii_uppercase + string.digits
    literal = "AK" + "IA" + "".join(random.SystemRandom().choice(alphabet) for _ in range(16))  # AWS key format
    store = sqlite_store(tmp_path)
    echoing_output = (BIG_PREFIX + "order,1,shipped\n" * 500 + f"debug: authorization={literal}\n").encode()
    spill = ToolOutputSpiller(store, threshold_bytes=1024).spill_if_large(
        tenant_id=TENANT_A, run_id="run-1", tool_id="vendor.orders.read", agent_id="agent-1", content=echoing_output
    )
    assert spill.reference is not None
    assert literal.encode() not in store.get(TENANT_A, spill.reference.artifact_id).data
    assert literal.encode() not in (tmp_path / "artifacts.sqlite3").read_bytes()
