# © Artur Czarnecki. All rights reserved.

"""XD-11 (All <-> Observability/Trace): every mandatory operation emits canonical trace or evidence.

Inventory-driven. ``INVENTORY`` lists the mandatory operations the XD scenarios rely on, one entry per
operation with the scenario that exercises it and a driver that performs the operation on the real
component and returns the evidence it emitted. The assertions are generic: the evidence must be
non-empty and carry the canonical identity fields (tenant plus one correlation key). The coverage
tests tie the inventory to the scenario modules in both directions, so a new scenario or a new
mandatory operation without evidence fails here.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from intergrax.applications._shared.security_audit_trail_wiring import resolve_security_audit_trail_wiring
from intergrax.background_tasks.definition import TaskDefinition
from intergrax.background_tasks.events import TaskEvent
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
from intergrax.runtime.nexus.artifacts.governed_access import ArtifactReadRequest
from intergrax.runtime.nexus.artifacts.retention import ArtifactRetentionSweeper
from intergrax.runtime.sandbox.network_egress import canonicalize_network_egress_allowlist
from intergrax.tools.execution_models import ToolExecutionResult
from pydantic import BaseModel
from tests.qualification.pdc_x.d1 import _d1_harness as h
from tests.qualification.pdc_x.d2._support import FakeAdapter
from tests.qualification.pdc_x.d4._support import (
    JournalRuntime,
    RecordingPublisher,
    build_engine,
    definition_of,
    make_profile,
)
from tests.qualification.pdc_x.d5._support import build_run
from tests.qualification.pdc_x.d6._support import (
    CountingSecretsStore,
    generated_secret,
    open_evidence_plane,
    product_env,
)
from tests.qualification.pdc_x.d8._d8_support import (
    T0,
    TENANT_A,
    TENANT_B,
    MutableClock,
    make_artifact,
    make_context,
    make_governance,
    make_port,
    reader_grants,
    sqlite_store,
)
from tests.qualification.pdc_x.xd._support import build_sqlite_agent_stack, build_sqlite_worker

pytestmark = [pytest.mark.integration]

XD_DIR = Path(__file__).resolve().parent
_IDENTITY_KEYS = ("run_id", "execution_id", "trace_id", "instance_id", "task_id", "grant_id", "revision_id", "ref")


@dataclass(frozen=True)
class Evidence:
    """One canonical evidence record reduced to its identity fields."""

    kind: str
    tenant: str | None
    correlation: dict[str, str]


@dataclass(frozen=True)
class Operation:
    op_id: str
    scenarios: tuple[str, ...]
    driver: Callable[[Path], Awaitable[list[Evidence]]]


class _Output(BaseModel):
    ok: bool = True


# ----------------------------------------------------------------------- drivers


async def _worker_task(tmp_path: Path) -> list[Evidence]:
    collected: list[TaskEvent] = []

    class _Collector:
        def emit(self, event: TaskEvent) -> None:
            collected.append(event)

    def handler(*, tenant_id, run_id, payload, idempotency_key, execution_identity):  # type: ignore[no-untyped-def]
        return ToolExecutionResult.ok(_Output())

    worker = build_sqlite_worker(
        tmp_path,
        [TaskDefinition(task_name="xd11.task", payload_schema=dict, handler=handler)],
        event_emitter=_Collector(),
    )
    result = worker.process_request(
        TaskRequest(tenant_id="tenant-xd11", run_id="run-xd11", task_name="xd11.task", payload=b"{}"),
        task_id="run-xd11",
    )
    assert result.status is TaskStatus.SUCCEEDED
    return [
        Evidence(
            kind=event.name.value,
            tenant=event.tenant_id,
            correlation={"run_id": event.run_id, "task_id": event.task_id},
        )
        for event in collected
    ]


async def _host_lifecycle(tmp_path: Path) -> list[Evidence]:
    publisher = RecordingPublisher()
    profile = make_profile(runtime=JournalRuntime())
    engine, _, _ = build_engine(definition_of(profile), tmp_path / "host", publisher=publisher)
    await engine.start()
    await engine.stop(reason_code="xd11.done")
    return [
        Evidence(
            kind=event.event_type.value,
            tenant=None,  # the host is not tenant-scoped; its identity is application + instance + digest
            correlation={
                "instance_id": event.instance_id,
                "application_id": event.application_id,
                **(
                    {"profile_digest": str(event.payload["profile_digest"])}
                    if "profile_digest" in event.payload
                    else {}
                ),
            },
        )
        for event in publisher.events
    ]


def _governance_evidence(events) -> list[Evidence]:  # type: ignore[no-untyped-def]
    return [
        Evidence(
            kind=f"{event.capability}:{event.decision.value}",
            tenant=event.tenant_id,
            correlation={"run_id": str(event.run_id), "execution_id": str(event.execution_id), "agent": event.agent_id},
        )
        for event in events
    ]


async def _artifact_read_and_deny(tmp_path: Path) -> list[Evidence]:
    store = sqlite_store(tmp_path)
    store.put(make_artifact(tenant_id=TENANT_A))
    harness = make_governance(granted=reader_grants(tenant_id=TENANT_A))
    port, _ = make_port(store, harness=harness)
    request = ArtifactReadRequest(
        artifact_class=ArtifactClass.EXECUTION_EVIDENCE, tenant_id=TENANT_A, artifact_id="art-1"
    )
    port.read(make_context(tenant_id=TENANT_A), request)  # allowed read
    foreign = ArtifactReadRequest(
        artifact_class=ArtifactClass.EXECUTION_EVIDENCE, tenant_id=TENANT_B, artifact_id="art-1"
    )
    try:
        port.read(make_context(tenant_id=TENANT_A), foreign)  # denied read
    except Exception:  # noqa: BLE001 - the denial itself is the operation under test
        pass
    return _governance_evidence(harness.sink.events)


async def _artifact_ttl_delete(tmp_path: Path) -> list[Evidence]:
    store = sqlite_store(tmp_path)
    clock = MutableClock()
    store.put(
        make_artifact(
            artifact_id="ephemeral",
            retention_class=ArtifactRetentionClass.EPHEMERAL,
            expires_at_utc=T0 + timedelta(hours=1),
        )
    )
    harness = make_governance()
    clock.advance(hours=2)
    ArtifactRetentionSweeper(store, clock=clock, audit=harness.recorder).sweep(
        TENANT_A, context=make_context(tenant_id=TENANT_A, agent_id="artifact_retention_sweep")
    )
    return _governance_evidence(harness.sink.events)


class _Allow:
    def admit(self, grant, scope):  # type: ignore[no-untyped-def]
        return CredentialScopeAdmissionDecision.ALLOW


async def _credential_use(tmp_path: Path) -> list[Evidence]:
    hosts = canonicalize_network_egress_allowlist(("https://api.vendor.com:443",))
    path = "secrets/tenant-a/vendor-api"
    grant = CredentialUseGrant(
        grant_id="grant-xd11",
        credential_ref=CredentialRef.from_secret_path(provider_id="vendor_api", secret_path=path, tenant_id=TENANT_A),
        tenant_id=TENANT_A,
        provider_id="vendor_api",
        integration_id="vendor_api:issue_tracker",
        operation="vendor.orders.read",
        execution_id="exec-xd11",
        target_scope=hosts,
        expires_at=datetime.now(UTC) + timedelta(minutes=10),
    )
    scope = CredentialUseScope(
        tenant_id=TENANT_A,
        provider_id="vendor_api",
        integration_id="vendor_api:issue_tracker",
        operation="vendor.orders.read",
        execution_id="exec-xd11",
        target_scope=hosts,
    )
    broker = ScopedCredentialBroker(
        resolver=SecretsStoreCredentialResolver(CountingSecretsStore({path: generated_secret()})), admission=_Allow()
    )
    evidence = broker.resolve_scoped(grant, scope).use_evidence
    return [
        Evidence(
            kind="credential.use",
            tenant=evidence.tenant_id,
            correlation={"execution_id": evidence.execution_id, "grant_id": evidence.grant_id},
        )
    ]


async def _security_audit_trail(tmp_path: Path) -> list[Evidence]:
    db = tmp_path / "evidence.db"
    audit = resolve_security_audit_trail_wiring(product_env(), evidence_persistence=open_evidence_plane(db))
    assert audit.trail is not None
    audit.trail.append(
        tenant_id=TENANT_A, action="xd11.audited", actor_id="svc", resource="resource-1", metadata={"grant_id": "g-1"}
    )
    with sqlite3.connect(db) as connection:
        dump = "\n".join(connection.iterdump())
    present = "xd11.audited" in dump and TENANT_A in dump
    return [Evidence(kind="security.audit", tenant=TENANT_A, correlation={"ref": "resource-1"})] if present else []


async def _agent_lifecycle(tmp_path: Path) -> list[Evidence]:
    stack = build_sqlite_agent_stack(tmp_path / "agents.sqlite3")
    first = h.release("1.0.0")
    stack.register(first)
    installed = stack.service.install_agent(
        application_id=h.APP, application_environment_id=h.ENV, request=first.install_request(), principal=h.principal()
    )
    h.build_revision(stack, "rev-xd11")
    activated = h.activate_revision(stack, "rev-xd11")
    types = (*installed.audit_event_types, *activated.audit_event_types)
    return [
        Evidence(kind=kind, tenant=h.TENANT, correlation={"revision_id": "rev-xd11", "ref": h.APP}) for kind in types
    ]


async def _evaluation_run(tmp_path: Path) -> list[Evidence]:
    run = build_run()
    return [
        Evidence(
            kind="eval.run",
            tenant=run.tenant_id,
            correlation={"ref": run.run_id, "inputs_digest": run.inputs_digest, "result_digest": run.result_digest},
        )
    ]


async def _context_assembly(tmp_path: Path) -> list[Evidence]:
    from intergrax.context.contracts import ContextFragmentSource
    from intergrax.context.registry import ContextPluginRegistry
    from intergrax.runtime.nexus.context.context_engine import DefaultNexusContextEngine
    from tests.qualification.pdc_x.d2._support import (
        FragmentProvider,
        assembly_request,
        make_fragment,
        provider_context,
    )

    registry = ContextPluginRegistry()
    registry.add_provider(
        FragmentProvider(
            "xd11.context",
            {ContextFragmentSource.SYSTEM_INSTRUCTIONS},
            lambda: [make_fragment("frag-1", ContextFragmentSource.SYSTEM_INSTRUCTIONS, "instruction", mandatory=True)],
        )
    )
    assembled = await DefaultNexusContextEngine(registry=registry).assemble(
        assembly_request(budget_tokens=2000), provider_ctx=provider_context(FakeAdapter(), "question")
    )
    assert assembled.provider_outcomes and assembled.provider_outcomes[0].status == "success"
    return [
        Evidence(
            kind="context.fragment_provenance",
            tenant=None,  # fragment provenance carries provider and content identity, not the tenant
            correlation={"ref": item.fragment_id, "provider_id": item.provider_id, "content_hash": item.content_hash},
        )
        for item in assembled.provenance
    ]


INVENTORY: tuple[Operation, ...] = (
    Operation("agent.lifecycle", ("XD-01", "XD-04"), _agent_lifecycle),
    Operation("context.assembly", ("XD-01", "XD-02", "XD-07"), _context_assembly),
    Operation("artifact.read_and_deny", ("XD-02", "XD-09", "XD-10"), _artifact_read_and_deny),
    Operation("artifact.ttl_delete", ("XD-09", "XD-10"), _artifact_ttl_delete),
    Operation("host.lifecycle", ("XD-04", "XD-05"), _host_lifecycle),
    Operation("worker.task", ("XD-05", "XD-06"), _worker_task),
    Operation("credential.use", ("XD-06", "XD-10"), _credential_use),
    Operation("security.audit_trail", ("XD-03", "XD-06", "XD-10"), _security_audit_trail),
    Operation("eval.run", ("XD-08",), _evaluation_run),
)

# Scenario modules whose operations are evidence-checked elsewhere or are themselves observability checks.
_SELF_COVERING = {"XD-11": "this module", "XD-12": "ENT-AUDIT selection is evidence by construction (manifest run)"}


# ----------------------------------------------------------------------- generic assertions


@pytest.mark.parametrize("operation", INVENTORY, ids=[op.op_id for op in INVENTORY])
async def test_xd11_every_mandatory_operation_emits_canonical_evidence(operation: Operation, tmp_path: Path) -> None:
    evidence = await operation.driver(tmp_path)
    assert evidence, f"{operation.op_id} emitted no trace or evidence"
    for record in evidence:
        assert record.kind, f"{operation.op_id}: evidence without a kind"
        if record.tenant is not None:
            assert record.tenant.strip(), f"{operation.op_id}: blank tenant on {record.kind}"
        assert any(value and value != "None" for value in record.correlation.values()), (
            f"{operation.op_id}: {record.kind} carries no correlation identity"
        )


async def test_xd11_worker_evidence_covers_the_full_task_lifecycle(tmp_path: Path) -> None:
    kinds = {record.kind for record in await _worker_task(tmp_path)}
    assert {"task.dispatched", "task.started", "task.succeeded", "task.result_stored"} <= kinds


async def test_xd11_host_evidence_covers_start_and_stop_with_the_profile_digest(tmp_path: Path) -> None:
    records = await _host_lifecycle(tmp_path)
    kinds = {record.kind for record in records}
    assert {"hosting.application.starting", "hosting.application.stopping", "hosting.application.stopped"} <= kinds
    assert any("profile_digest" in record.correlation for record in records)


async def test_xd11_governance_evidence_records_both_the_allow_and_the_deny(tmp_path: Path) -> None:
    kinds = {record.kind for record in await _artifact_read_and_deny(tmp_path)}
    assert any(kind.endswith(":allow") or kind.endswith(":ALLOW") for kind in kinds)
    assert any(kind.endswith(":deny") or kind.endswith(":DENY") for kind in kinds)


# ----------------------------------------------------------------------- inventory <-> scenario coverage


def test_xd11_every_scenario_module_is_covered_by_an_inventoried_operation() -> None:
    modules = sorted(path.name for path in XD_DIR.glob("test_xd_[0-9][0-9]_*.py"))
    scenarios = {"XD-" + name.split("_")[2] for name in modules}
    covered = {scenario for op in INVENTORY for scenario in op.scenarios} | set(_SELF_COVERING)
    assert scenarios <= covered, f"scenarios without an inventoried operation: {sorted(scenarios - covered)}"


def test_xd11_every_inventoried_scenario_has_a_module_and_ids_are_unique() -> None:
    modules = {"XD-" + path.name.split("_")[2] for path in XD_DIR.glob("test_xd_[0-9][0-9]_*.py")}
    claimed = {scenario for op in INVENTORY for scenario in op.scenarios}
    assert claimed <= modules, f"inventory names scenarios with no module: {sorted(claimed - modules)}"
    ids = [op.op_id for op in INVENTORY]
    assert len(ids) == len(set(ids))
    assert json.dumps(ids)  # ids are plain serializable names
