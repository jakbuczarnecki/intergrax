"""IA-D8 adversarial probes (read-only against product code; gitignored scratch)."""
from __future__ import annotations

import tempfile
import traceback
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

from tests.qualification.pdc_x.d8._d8_support import (
    T0, TENANT_A, TENANT_B, FakeObjectStorage, MutableClock, make_artifact, make_context,
    make_governance, make_port, reader_grants, sqlite_store,
)
from intergrax.contracts.artifact_ref import (
    ArtifactClass, ArtifactLineage, ArtifactParent, ArtifactProvenance, ArtifactRetentionClass, ArtifactRef,
    ArtifactScope, DurableArtifactRef, compute_artifact_digest, is_valid_artifact_digest,
    reconstruct_artifact_lineage,
)
from intergrax.runtime.nexus.artifacts.governed_access import (
    ArtifactAccessDenied, ArtifactReadRequest,
)
from intergrax.runtime.nexus.artifacts.retention import ArtifactRetentionSweeper
from intergrax.runtime.nexus.artifacts.in_memory_artifact_store import InMemoryArtifactStore

R = {}


def probe(name):
    def deco(fn):
        try:
            R[name] = fn()
        except Exception as exc:  # noqa
            R[name] = f"EXC {type(exc).__name__}: {exc}"
        print(f"[{name}] -> {R[name]}")
        return fn
    return deco


@probe("P1_create_artifact_gateway")
def _():
    from intergrax.runtime.execution.agent_runtime_context import AgentRuntimeContext
    import inspect
    sig = inspect.signature(AgentRuntimeContext.create_artifact)
    from intergrax.runtime.nexus.artifacts.models import Artifact
    try:
        Artifact(artifact_id="x", run_id="r", step_id=None, kind="k", mime_type="t",
                 created_at_utc=T0, data=b"d", size_bytes=1)
    except TypeError as e:
        return f"Artifact(...) as built in create_artifact raises TypeError: {e}"
    return "no error"


@probe("P2_digest_trailing_newline")
def _():
    good = "sha256:" + "a" * 64
    return {"plain": is_valid_artifact_digest(good), "with_nl": is_valid_artifact_digest(good + "\n")}


@probe("P3_retention_class_downgrade")
def _():
    from intergrax.contracts.artifact_ref import artifact_ref_from_payload
    out = {}
    for val in ("audit_evidence", "AUDIT_EVIDENCE", "audit-evidence", "evidence"):
        ref = artifact_ref_from_payload({"retention_class": val, "checksum": "sha256:" + "b" * 64, "size_bytes": 3},
                                        run_id="r", trace_id="t", agent_id="a")
        d = ref.to_durable_ref(scope=ArtifactScope(tenant_id="t"), lineage=ArtifactLineage(run_id="r"))
        out[val] = d.retention_class.value
    return out


@probe("P4_storage_tools_bypass")
def _():
    import base64
    from intergrax.collaborative_work.content_storage import (
        ObjectStorageArtifactContentStore, StoreArtifactContentRequest, artifact_content_object_key,
    )
    from intergrax.tools.providers.storage import service as svc
    from intergrax.tools.providers.storage.contracts import StorageDeleteInput, StorageGetInput, StorageExistsInput
    os_ = FakeObjectStorage()
    cs = ObjectStorageArtifactContentStore(os_)
    ref = cs.put(StoreArtifactContentRequest(tenant_id="tenant-victim", workspace_id="ws-v", body=b"secret business", media_type="text/plain"))
    key = artifact_content_object_key("tenant-victim", "ws-v", ref)

    class Ctx:  # ToolWiringContext double carrying only object_storage
        object_storage = os_
    got = svc.storage_get(Ctx(), StorageGetInput(key=key))
    ex = svc.storage_exists(Ctx(), StorageExistsInput(key=key))
    svc.storage_delete(Ctx(), StorageDeleteInput(key=key))
    return {"get_found": got.found, "body_len": len(base64.b64decode(got.body_base64 or "")), "exists": ex.exists, "after_delete_in_store": key in os_.objects}


@probe("P5_read_after_expiry_unswept")
def _():
    with tempfile.TemporaryDirectory() as d:
        store = sqlite_store(Path(d))
        clock = MutableClock()
        art = make_artifact(retention_class=ArtifactRetentionClass.EPHEMERAL, expires_at_utc=T0 + timedelta(seconds=10))
        store.put(art)
        port, h = make_port(store, clock=clock)
        clock.advance(days=30)
        res = port.read(make_context(), ArtifactReadRequest(artifact_class=ArtifactClass.EXECUTION_EVIDENCE, tenant_id=TENANT_A, artifact_id="art-1"))
        return f"expired 30d ago, read succeeded: {len(res.content)} bytes"


@probe("P6_workspace_scope_optional")
def _():
    store = InMemoryArtifactStore()
    store.put(make_artifact())  # workspace ws-1
    port, h = make_port(store)
    ok = port.read(make_context(), ArtifactReadRequest(artifact_class=ArtifactClass.EXECUTION_EVIDENCE, tenant_id=TENANT_A, artifact_id="art-1", workspace_id=None))
    try:
        port.read(make_context(), ArtifactReadRequest(artifact_class=ArtifactClass.EXECUTION_EVIDENCE, tenant_id=TENANT_A, artifact_id="art-1", workspace_id="ws-OTHER"))
        other = "allowed"
    except ArtifactAccessDenied as e:
        other = f"denied {e.reason_code}"
    return {"no_workspace_in_request": "allowed ws=" + str(ok.ref.scope.workspace_id), "wrong_workspace": other}


@probe("P7_sweep_starvation_by_protected")
def _():
    store = InMemoryArtifactStore()
    clock = MutableClock()
    # 5 legal-hold artifacts expire first, limit=5 -> the unprotected one never gets swept
    for i in range(5):
        store.put(make_artifact(artifact_id=f"held-{i}", retention_class=ArtifactRetentionClass.EPHEMERAL, expires_at_utc=T0 - timedelta(days=2, seconds=i)))
        store.place_legal_hold(TENANT_A, f"held-{i}", placed_by="legal", reason="case")
    store.put(make_artifact(artifact_id="plain", retention_class=ArtifactRetentionClass.EPHEMERAL, expires_at_utc=T0 - timedelta(days=1)))
    sw = ArtifactRetentionSweeper(store, clock=clock)
    r1 = sw.sweep(TENANT_A, limit=5)
    r2 = sw.sweep(TENANT_A, limit=5)
    return {"first": (r1.examined, r1.deleted_ids, len(r1.protected_ids)), "second_deleted": r2.deleted_ids, "plain_still_there": True if store.get(TENANT_A, "plain") else False}


@probe("P8_poisoned_row_aborts_sweep")
def _():
    import sqlite3
    with tempfile.TemporaryDirectory() as d:
        store = sqlite_store(Path(d))
        clock = MutableClock()
        store.put(make_artifact(artifact_id="a-bad", retention_class=ArtifactRetentionClass.EPHEMERAL, expires_at_utc=T0 - timedelta(days=1), data=b"orig"))
        store.put(make_artifact(artifact_id="a-good", retention_class=ArtifactRetentionClass.EPHEMERAL, expires_at_utc=T0 - timedelta(hours=1), data=b"fine"))
        conn = sqlite3.connect(str(Path(d) / "artifacts.sqlite3"))
        conn.execute("UPDATE execution_artifacts SET data=? WHERE artifact_id='a-bad'", (b"tampered",))
        conn.commit(); conn.close()
        sw = ArtifactRetentionSweeper(store, clock=clock)
        try:
            rep = sw.sweep(TENANT_A)
            return f"sweep ok deleted={rep.deleted_ids}"
        except Exception as e:
            good_alive = True
            try:
                store.get(TENANT_A, "a-good")
            except Exception:
                good_alive = False
            return f"sweep aborted {type(e).__name__}; a-good still stored={good_alive}"


@probe("P9_lineage_cross_tenant_parent")
def _():
    pd = DurableArtifactRef(artifact_id="p", artifact_class=ArtifactClass.EXECUTION_EVIDENCE, type="t", uri="u",
                            content_digest=compute_artifact_digest(b"p"), size_bytes=1,
                            scope=ArtifactScope(tenant_id="tenant-OTHER"), producer=ArtifactProvenance(created_by_agent_id="a"),
                            lineage=ArtifactLineage(run_id="r0"))
    ch = DurableArtifactRef(artifact_id="c", artifact_class=ArtifactClass.EXECUTION_EVIDENCE, type="t", uri="u",
                            content_digest=compute_artifact_digest(b"c"), size_bytes=1,
                            scope=ArtifactScope(tenant_id="tenant-A"), producer=ArtifactProvenance(created_by_agent_id="a"),
                            lineage=ArtifactLineage(run_id="r1", parent_artifacts=(ArtifactParent(artifact_id="p", content_digest=pd.content_digest),)))
    chain = reconstruct_artifact_lineage("c", [pd, ch])
    return [f"{x.scope.tenant_id}/{x.artifact_id}" for x in chain]


@probe("P10_unhandled_error_not_audited")
def _():
    store = InMemoryArtifactStore()
    store.put(make_artifact())
    port, h = make_port(store)
    n0 = len(h.sink.events) if hasattr(h.sink, "events") else -1
    # class not registered -> no_source (audited). Now force an unexpected source error:
    class Boom:
        def read(self, request):
            raise RuntimeError("backend down")
    port._sources[ArtifactClass.EXECUTION_EVIDENCE] = Boom()
    try:
        port.read(make_context(), ArtifactReadRequest(artifact_class=ArtifactClass.EXECUTION_EVIDENCE, tenant_id=TENANT_A, artifact_id="art-1"))
    except RuntimeError:
        pass
    evs = list(getattr(h.sink, "events", []) or [])
    last = evs[-1] if evs else None
    return {"events_total": len(evs), "last_capability": getattr(getattr(last, "request", None) or last, "capability", None) if last else None, "outcome_event_for_failure": any("outcome" in str(getattr(e, "capability", "")) for e in evs)}


@probe("P11_evidence_via_late_version")
def _():
    store = InMemoryArtifactStore()
    store.put(make_artifact(artifact_id="ev", retention_class=ArtifactRetentionClass.AUDIT_EVIDENCE))
    store.put(make_artifact(artifact_id="ev", retention_class=ArtifactRetentionClass.EPHEMERAL, expires_at_utc=T0 - timedelta(days=1), data=b"v2"))
    try:
        store.delete_artifact(TENANT_A, "ev")
        return "deleted (BAD)"
    except Exception as e:
        return f"refused: {type(e).__name__}"


@probe("P12_nonuniq_tenant_empty")
def _():
    store = InMemoryArtifactStore()
    a = make_artifact(tenant_id="")
    store.put(a)
    return "empty tenant_id artifact accepted by store"


@probe("P13_legal_hold_holder_overwritten_sqlite")
def _():
    with tempfile.TemporaryDirectory() as d:
        import sqlite3
        store = sqlite_store(Path(d))
        store.put(make_artifact())
        store.place_legal_hold(TENANT_A, "art-1", placed_by="legal-counsel", reason="litigation")
        store.place_legal_hold(TENANT_A, "art-1", placed_by="other-agent", reason="x")
        conn = sqlite3.connect(str(Path(d) / "artifacts.sqlite3"))
        row = conn.execute("select placed_by, reason from execution_artifact_legal_holds").fetchall()
        conn.close()
        return row


@probe("P14_inmemory_hold_metadata_dropped")
def _():
    store = InMemoryArtifactStore()
    store.put(make_artifact())
    store.place_legal_hold(TENANT_A, "art-1", placed_by="legal", reason="case")
    return f"in-memory holds attr: {store._holds}"

print("DONE")
