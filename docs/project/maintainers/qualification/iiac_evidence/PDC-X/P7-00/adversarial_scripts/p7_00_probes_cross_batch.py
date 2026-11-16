"""P7-00 throw-away probes #2: confirm or refute subagent-reported candidate findings (auditor-owned, read-only)."""

from __future__ import annotations

import json
from datetime import timedelta

results: dict[str, object] = {}


def probe(name):
    def deco(fn):
        try:
            results[name] = fn()
        except BaseException as exc:  # noqa: BLE001
            results[name] = {"probe_error": f"{type(exc).__name__}: {exc}"}
        return fn

    return deco


@probe("X1_artifact_read_without_workspace_id_crosses_workspace")
def _():
    from intergrax.contracts.artifact_ref import ArtifactClass
    from intergrax.runtime.nexus.artifacts.governed_access import ArtifactAccessDenied, ArtifactReadRequest
    from intergrax.runtime.nexus.artifacts.in_memory_artifact_store import InMemoryArtifactStore
    from tests.qualification.pdc_x.d8._d8_support import TENANT_A, make_artifact, make_context, make_port

    store = InMemoryArtifactStore()
    store.put(make_artifact())  # stored with workspace_id="ws-1"
    port, _h = make_port(store)
    out = {}
    for label, ws in {"no_workspace": None, "right_workspace": "ws-1", "wrong_workspace": "ws-OTHER"}.items():
        try:
            port.read(
                make_context(),
                ArtifactReadRequest(
                    artifact_class=ArtifactClass.EXECUTION_EVIDENCE,
                    tenant_id=TENANT_A,
                    artifact_id="art-1",
                    workspace_id=ws,
                ),
            )
            out[label] = "READ_OK"
        except ArtifactAccessDenied as exc:
            out[label] = f"denied:{exc.reason_code}"
    return out


@probe("X2_governance_port_returning_deny_without_raising_is_treated_as_allow")
def _():
    from intergrax.contracts.agent_runtime_governance import ToolAuthorizationDecision, ToolAuthorizationDecisionState
    from intergrax.contracts.artifact_ref import ArtifactClass
    from intergrax.runtime.nexus.artifacts.governed_access import ArtifactReadRequest, GovernedArtifactReadPort
    from intergrax.runtime.nexus.artifacts.in_memory_artifact_store import InMemoryArtifactStore
    from intergrax.runtime.nexus.artifacts.sources import ExecutionArtifactSource
    from tests.qualification.pdc_x.d8._d8_support import (
        TENANT_A,
        MutableClock,
        make_artifact,
        make_context,
        make_governance,
    )

    class DenyingPort:
        def authorize_tool(self, request, *, policy_context=None):
            return ToolAuthorizationDecision(decision=ToolAuthorizationDecisionState.DENY, reason="nope")

    h = make_governance()
    store = InMemoryArtifactStore()
    store.put(make_artifact())
    port = GovernedArtifactReadPort(
        governance=DenyingPort(),
        audit=h.recorder,
        sources={ArtifactClass.EXECUTION_EVIDENCE: ExecutionArtifactSource(store)},
        clock=MutableClock(),
    )
    try:
        port.read(
            make_context(),
            ArtifactReadRequest(artifact_class=ArtifactClass.EXECUTION_EVIDENCE, tenant_id=TENANT_A, artifact_id="art-1"),
        )
        return "READ_OK_DESPITE_DENY_DECISION"
    except Exception as exc:  # noqa: BLE001
        return f"refused:{type(exc).__name__}"


@probe("X3_retention_sweep_without_audit_deletes_silently_and_delete_race_destroys_new_version")
def _():
    from intergrax.runtime.nexus.artifacts.in_memory_artifact_store import InMemoryArtifactStore
    from intergrax.runtime.nexus.artifacts.retention import ArtifactRetentionSweeper
    from tests.qualification.pdc_x.d8._d8_support import T0, TENANT_A, make_artifact

    out = {}
    store = InMemoryArtifactStore()
    store.put(make_artifact(expires_at_utc=T0 - timedelta(hours=1)))
    rep = ArtifactRetentionSweeper(store, clock=lambda: T0).sweep(TENANT_A)
    out["sweep_without_audit"] = {"deleted": list(rep.deleted_ids), "audit_recorder_configured": False}
    store2 = InMemoryArtifactStore()
    store2.put(make_artifact(expires_at_utc=T0 - timedelta(hours=1)))
    expired = store2.list_expired(TENANT_A, T0, limit=10)
    store2.put(make_artifact(data=b"v2-non-expiring", expires_at_utc=None))  # concurrent writer
    try:
        store2.delete_artifact(TENANT_A, "art-1")
        survivors = store2.list_versions(TENANT_A, "art-1")
    except KeyError:
        survivors = []
    out["race"] = {"selected_as_expired": len(list(expired)), "versions_after_delete": len(survivors)}
    return out


@probe("X4_skill_upgrade_baseexception_leaves_skill_unregistered")
def _():
    from tests.qualification.pdc_x.d1.test_d1_skill_lifecycle import (
        _HOST,
        _SKILL,
        _installed,
        _Materializer,
        _resolution,
    )

    class Boom(BaseException):
        pass

    class BoomMat(_Materializer):
        def materialize(self, resolution):
            raise Boom()

    class ExcMat(_Materializer):
        def materialize(self, resolution):
            raise RuntimeError("x")

    lc = _installed()
    before = lc.registry.has(_SKILL)
    try:
        lc.upgrade_skill(
            operation_id="op-u",
            host_profile_id=_HOST,
            resolved=_resolution("1.1.0", "b"),
            materializer=BoomMat(lc.registry),
        )
    except Boom:
        pass
    lc2 = _installed()
    lc2.upgrade_skill(
        operation_id="op-u2",
        host_profile_id=_HOST,
        resolved=_resolution("1.1.0", "b"),
        materializer=ExcMat(lc2.registry),
    )
    return {
        "registered_before": before,
        "registered_after_BaseException": lc.registry.has(_SKILL),
        "registered_after_Exception_control": lc2.registry.has(_SKILL),
    }


@probe("X5_skill_upgrade_prerelease_and_build_tag_ordering")
def _():
    from intergrax.skills.host_lifecycle import _upgrade_denial

    return {
        "1.2.3->1.2.4-rc.1": _upgrade_denial("1.2.3", "1.2.4-rc.1"),
        "1.2.3->1.2.3+evil": _upgrade_denial("1.2.3", "1.2.3+evil"),
        "1.2.3->1.2.3-rc.1": _upgrade_denial("1.2.3", "1.2.3-rc.1"),
    }


@probe("X6_recurrence_schedule_id_has_no_tenant_component")
def _():
    import inspect
    from datetime import UTC, datetime

    from intergrax.runtime.long_running.recurrence import occurrence_schedule_id

    when = datetime(2026, 12, 1, 9, 0, tzinfo=UTC)
    return {
        "params": list(inspect.signature(occurrence_schedule_id).parameters),
        "id": occurrence_schedule_id("daily-report", when),
    }


@probe("X7_reindex_hook_exception_propagates_from_freshness_assessment")
def _():
    import inspect

    from intergrax.rag.governance import embedding_version_policy as evp

    return {"hook_loop_has_try": "try:" in inspect.getsource(evp._notify_reindex_needed)}


@probe("X8_lab_profile_signs_boundary_events_without_key")
def _():
    from intergrax.applications._shared.attestation_runtime_bridge import build_boundary_event_buffer
    from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile

    env = ApplicationEnvironmentProfile.lab_defaults()
    prof = env.execution_boundary_export_profile
    out = {"export_profile_present": prof is not None, "execution_mode": str(env.execution_mode)}
    if prof is not None:
        env.execution_boundary_export_profile = prof.model_copy(update={"enabled": True, "host_signing_enabled": True})
        buf = build_boundary_event_buffer(env, environ={})
        out["lab_buffer_built_without_env_key"] = buf is not None
    return out


print(json.dumps(results, indent=2, default=str, sort_keys=True))
