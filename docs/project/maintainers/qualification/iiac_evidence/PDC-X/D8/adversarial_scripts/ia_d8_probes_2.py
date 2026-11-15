from __future__ import annotations
import os, tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tests.qualification.pdc_x.d8._d8_support import *
from intergrax.contracts.agent_runtime_governance import ToolAuthorizationDecision, ToolAuthorizationDecisionState
from intergrax.contracts.artifact_ref import ArtifactClass, ArtifactRetentionClass
from intergrax.runtime.nexus.artifacts.governed_access import ArtifactReadRequest, GovernedArtifactReadPort
from intergrax.runtime.nexus.artifacts.in_memory_artifact_store import InMemoryArtifactStore
from intergrax.runtime.nexus.artifacts.retention import ArtifactRetentionSweeper, make_retention_sweep_handler
from intergrax.runtime.nexus.artifacts.sources import ExecutionArtifactSource
R = {}
def probe(name):
    def deco(fn):
        try: R[name] = fn()
        except Exception as e: R[name] = f"EXC {type(e).__name__}: {e}"
        print(f"[{name}] -> {R[name]}")
    return deco

@probe("F02_toctou_new_nonexpiring_version_deleted")
def _():
    for label, store in (("memory", InMemoryArtifactStore()), ("sqlite", sqlite_store(Path(tempfile.mkdtemp())))):
        store.put(make_artifact(retention_class=ArtifactRetentionClass.EPHEMERAL, expires_at_utc=T0 - timedelta(hours=1)))
        cand = store.list_expired(TENANT_A, T0)
        store.put(make_artifact(data=b"v2 keep me", retention_class=ArtifactRetentionClass.DEFAULT))  # v2 non-expiring
        # sweeper's decision path after selection (as in sweep()): delete_artifact deletes all versions
        store.delete_artifact(TENANT_A, cand[0].artifact_id)
        left = list(store.list_versions(TENANT_A, "art-1"))
        print(label, "versions left:", len(left))
    return "see above"

@probe("F02_sweep_without_audit_deletes_silently")
def _():
    store = InMemoryArtifactStore()
    store.put(make_artifact(retention_class=ArtifactRetentionClass.EPHEMERAL, expires_at_utc=T0 - timedelta(hours=1)))
    rep = ArtifactRetentionSweeper(store, clock=MutableClock()).sweep(TENANT_A)
    return rep.deleted_ids

@probe("F10_permissive_governance_returning_deny")
def _():
    class Permissive:
        def authorize_tool(self, request):
            return ToolAuthorizationDecision(decision=ToolAuthorizationDecisionState.DENY, reason="no")
    store = InMemoryArtifactStore(); store.put(make_artifact())
    h = make_governance(granted=reader_grants())
    port = GovernedArtifactReadPort(governance=Permissive(), audit=h.recorder,
        sources={ArtifactClass.EXECUTION_EVIDENCE: ExecutionArtifactSource(store)})
    r = port.read(make_context(), ArtifactReadRequest(artifact_class=ArtifactClass.EXECUTION_EVIDENCE, tenant_id=TENANT_A, artifact_id="art-1"))
    return f"READ_OK_DESPITE_DENY ({len(r.content)} bytes)"

@probe("F28_naive_vs_aware_datetime_sweep")
def _():
    store = InMemoryArtifactStore()
    store.put(make_artifact(retention_class=ArtifactRetentionClass.EPHEMERAL, expires_at_utc=datetime(2020,1,1)))  # naive
    try:
        ArtifactRetentionSweeper(store, clock=MutableClock()).sweep(TENANT_A)
        return "ok"
    except Exception as e:
        return f"sweep killed: {type(e).__name__}: {e}"

@probe("F28_negative_limit")
def _():
    store = InMemoryArtifactStore()
    for i in range(3):
        store.put(make_artifact(artifact_id=f"e{i}", retention_class=ArtifactRetentionClass.EPHEMERAL, expires_at_utc=T0 - timedelta(hours=1+i)))
    s = ArtifactRetentionSweeper(store, clock=MutableClock())
    r = s.sweep(TENANT_A, limit=-1)
    sq = sqlite_store(Path(tempfile.mkdtemp()))
    for i in range(3):
        sq.put(make_artifact(artifact_id=f"e{i}", retention_class=ArtifactRetentionClass.EPHEMERAL, expires_at_utc=T0 - timedelta(hours=1+i)))
    r2 = ArtifactRetentionSweeper(sq, clock=MutableClock()).sweep(TENANT_A, limit=-1)
    return {"memory_examined_with_limit_-1": r.examined, "sqlite_examined_with_limit_-1": r2.examined}

@probe("F27_symlink_in_sandbox")
def _():
    from intergrax.runtime.nexus.orchestration import run_artifact_bundle_builder as b
    import inspect
    src = inspect.getsource(b._sandbox_artifact_refs)
    d = Path(tempfile.mkdtemp()); root = d / "root"; root.mkdir(); outside = d / "outside.txt"; outside.write_text("outside secret")
    try:
        os.symlink(outside, root / "link.txt")
    except OSError as e:
        return f"symlink creation unavailable on this host: {e}; code reads path.read_bytes() after is_file(): {'is_file' in src}"
    rows = [p for p in sorted(root.rglob("*")) if p.is_file()]
    return f"is_file() true for symlink to outside: {[p.name for p in rows]}; read_bytes -> {rows[0].read_bytes()!r}"

@probe("download_after_expiry_and_hold")
def _():
    return "n/a"

@probe("P_expired_download_missing_object_still_issued")
def _():
    from intergrax.collaborative_work.content_storage import ObjectStorageArtifactContentStore
    return "n/a"
print("DONE")
