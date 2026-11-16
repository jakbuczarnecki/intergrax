"""IA-D4 adversarial probes A (no product edits; everything in-process)."""
import asyncio, json, os, sys, tempfile, time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
out = {}

# P1: F-22 typo accepted; wide bind accepted
from intergrax.applications.contracts.environment_profile import HostDeploymentProfile
p = HostDeploymentProfile(business_agents_deploy_enabld=True)
out["P1_typo_accepted"] = {"flag": p.business_agents_deploy_enabld if hasattr(p, "business_agents_deploy_enabld") else None,
                           "typed_flag_value": p.business_agents_deploy_enabled, "extra": p.model_extra}
from intergrax.applications._shared.lkw_hybrid_daemon_wiring import LkwHybridDaemonDeployment
d = LkwHybridDaemonDeployment.from_host_deployment(HostDeploymentProfile(lkw_hybrid_daemon_enabled=True, lkw_daemon_bind_host="0.0.0.0"))
out["P1_wide_bind"] = d.lkw_daemon_bind_host
try:
    LkwHybridDaemonDeployment.from_host_deployment(HostDeploymentProfile(lkw_daemon_prot=1))
    out["P1_lkw_typo"] = "accepted"
except Exception as e:
    out["P1_lkw_typo"] = "rejected:" + type(e).__name__
try:
    HostDeploymentProfile(some_other_key={"a": 1})
    out["P1_nested"] = "accepted"
except Exception as e:
    out["P1_nested"] = "rejected:" + type(e).__name__

# P2: public API fingerprint blind spots
from pydantic import BaseModel, ConfigDict, Field
from intergrax.hosting.public_api import fingerprint_public_symbol
class A(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    x: int = Field(default=1, ge=1, le=5)
class B(BaseModel):
    model_config = ConfigDict(extra="allow")
    x: int = Field(default=1, ge=-100, le=500)
B.__name__ = "A"
out["P2_same_fingerprint_for_loosened_contract"] = fingerprint_public_symbol(A) == fingerprint_public_symbol(B)
import intergrax.hosting as hosting
from intergrax.hosting import HostedApplicationProfile
before = fingerprint_public_symbol(HostedApplicationProfile)
orig = dict(HostedApplicationProfile.model_config)
HostedApplicationProfile.model_config["extra"] = "allow"
after = fingerprint_public_symbol(HostedApplicationProfile)
HostedApplicationProfile.model_config.update(orig)
out["P2_profile_extra_forbid_to_allow_changes_fingerprint"] = before != after

# P3: PID reuse false conflict
from intergrax.hosting.instance.file_guard import FileHostedApplicationInstanceGuard, _LeaseMetadata
from intergrax.hosting.contracts.context import HostedApplicationProcessIdentity
from intergrax.hosting.contracts.policies import InstancePolicy
from intergrax.hosting.instance.contracts import HostedApplicationInstanceIdentity
from intergrax.hosting.errors import HostedApplicationInstanceConflictError
class Clock:
    def now(self): return datetime.now(UTC)
async def p3():
    tmp = Path(tempfile.mkdtemp(prefix="iad4_"))
    run = tmp / "run"; run.mkdir()
    pid = os.getpid()
    old_start = datetime(2020, 1, 1, tzinfo=UTC)
    meta = _LeaseMetadata(schema_version="1.0", application_id="probe_app", instance_id="old-instance", process_id=pid,
                          process_started_at=old_start, host_id=None, user_scope_id=None, profile_digest="sha256:x",
                          acquired_at=old_start, ownership_token="tok")
    (run / "probe_app.lock").write_bytes(meta.to_json_bytes())  # crashed owner left metadata; no OS lock held
    ident = HostedApplicationProcessIdentity(process_id=pid, started_at=datetime.now(UTC), host_id=None)
    g = FileHostedApplicationInstanceGuard(run_directory=run, instance_policy=InstancePolicy(), process_identity=ident, clock=Clock())
    try:
        res = await g.acquire(HostedApplicationInstanceIdentity(application_id="probe_app", instance_id="new-instance", profile_digest="sha256:y", process_identity=ident))
        await res.lease.release()
        return "acquired:" + str(res.classification)
    except HostedApplicationInstanceConflictError as e:
        return "conflict:" + str(e.snapshot.reason_code if hasattr(e, "snapshot") else e)
out["P3_pid_reuse_prior_metadata_with_live_pid_different_start"] = asyncio.run(p3())

# P5/P6: metrics through bridge: which survive, default policy exports nothing
from intergrax.hosting import ObservabilityHostedApplicationEventPublisher
from intergrax.hosting.metrics import HostedApplicationLifecycleMetric, lifecycle_metric_event
from intergrax.hosting.contracts.lifecycle import HostedApplicationLifecycleState
from intergrax.runtime.observability.export_boundary import InMemoryObservabilityExporter
from intergrax.runtime.observability.export_policy import ObservabilityExportPolicy
async def p5():
    exp = InMemoryObservabilityExporter()
    pub = ObservabilityHostedApplicationEventPublisher(exp, policy=ObservabilityExportPolicy(enabled=True))
    exp_default = InMemoryObservabilityExporter()
    pub_default = ObservabilityHostedApplicationEventPublisher(exp_default)
    for m in HostedApplicationLifecycleMetric:
        ev = lifecycle_metric_event(m, application_id="probe_app", instance_id="i1", lifecycle_state=HostedApplicationLifecycleState.READY, value=1, label="lbl")
        await pub.publish(ev); await pub_default.publish(ev)
    names = {}
    for env in exp.envelopes:
        a = env.sanitized_application_attributes.attributes if env.sanitized_application_attributes else {}
        names[a.get("hosting.metric_name")] = a.get("hosting.metric_label")
    return names, len(exp_default.envelopes)
names, default_count = asyncio.run(p5())
out["P5_enabled_bridge_metric_names_to_label"] = names
out["P6_default_publisher_exported_envelopes"] = default_count

# P7: run_bounded_phase with cancellation-swallowing work overruns
from intergrax.hosting.shutdown import run_bounded_phase, HostedApplicationGlobalShutdownBudget, SystemMonotonicClock
async def p7():
    clk = SystemMonotonicClock()
    budget = HostedApplicationGlobalShutdownBudget(deadline_monotonic=clk.monotonic() + 0.5, monotonic_clock=clk)
    async def stubborn():
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            await asyncio.sleep(2.0)
    t = time.monotonic()
    outcome = await run_bounded_phase(budget, 0.2, stubborn)
    return outcome.value, round(time.monotonic() - t, 2)
out["P7_bounded_phase_swallowed_cancel(outcome,elapsed_s; limit 0.2s)"] = asyncio.run(p7())
print(json.dumps(out, indent=1, default=str))
