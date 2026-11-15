"""IA-D4 adversarial probes B: engine-level (no product edits)."""
import asyncio, json, sys, tempfile, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from intergrax.hosting import (HostedApplicationComponentRegistration, HostedApplicationEventType,
                               HostedApplicationStartupError, HostedApplicationLifecycleState)
from intergrax.hosting.contracts.policies import ComponentFailureAction
from intergrax.hosting.engine.components import ComponentCoordinator  # noqa
from tests.qualification.pdc_x.d4._support import (OrderedComponent, JournalRuntime, RecordingPublisher, build_engine,
                                                   definition_of, make_profile)
out = {}

async def probe_post_runtime_rollback():
    j = []
    tmp = Path(tempfile.mkdtemp(prefix="iad4b_"))
    core = HostedApplicationComponentRegistration(component=OrderedComponent("core", j), required=True)
    post_ok = HostedApplicationComponentRegistration(component=OrderedComponent("post_ok", j),
                                                     failure_action=ComponentFailureAction.FAIL_HOST)
    post_bad = HostedApplicationComponentRegistration(component=OrderedComponent("post_bad", j, fail_start=True),
                                                      dependencies=("post_ok",),
                                                      failure_action=ComponentFailureAction.FAIL_HOST)
    prof = make_profile(runtime=JournalRuntime(j), components=(core, post_ok, post_bad))
    d = definition_of(prof)
    eng, pub, _ = build_engine(d, tmp)
    try:
        await eng.start()
        res = "started"
    except HostedApplicationStartupError:
        res = "startup_error"
    return {"pre": d.pre_runtime_component_ids, "post": d.post_runtime_component_ids, "journal": j, "result": res,
            "state": eng.lifecycle_snapshot().state.value}
out["P8_post_runtime_failure_rollback"] = asyncio.run(probe_post_runtime_rollback())

async def probe_cancel_during_start():
    j = []
    tmp = Path(tempfile.mkdtemp(prefix="iad4c_"))
    class Slow(OrderedComponent):
        async def start(self, context):
            self._journal.append(f"start:{self.component_id}")
            await asyncio.sleep(30)
    core = HostedApplicationComponentRegistration(component=OrderedComponent("core", j), required=True,
                                                  start_timeout_seconds=60.0, stop_timeout_seconds=5.0)
    slow = HostedApplicationComponentRegistration(component=Slow("slow", j), component_type_id="probe.slow", required=True, dependencies=("core",),
                                                  start_timeout_seconds=60.0, stop_timeout_seconds=5.0)
    prof = make_profile(runtime=JournalRuntime(j), components=(core, slow))
    d = definition_of(prof)
    eng, pub, _ = build_engine(d, tmp)
    t = asyncio.ensure_future(eng.start())
    await asyncio.sleep(0.5)
    t.cancel()
    await asyncio.gather(t, return_exceptions=True)
    diag = eng.diagnostics_snapshot()
    # can a new engine acquire the same lock in this process (lease leaked)?
    j2 = []
    eng2, _, _ = build_engine(definition_of(make_profile(runtime=JournalRuntime(j2))), tmp, instance_id="instance-002")
    try:
        await asyncio.wait_for(eng2.start(), 5)
        second = "second_engine_started"
        await eng2.stop()
    except Exception as e:
        second = "second_engine_rejected:" + type(e).__name__
    return {"journal": j, "state": eng.lifecycle_snapshot().state.value, "lease_acquired": diag.instance_lease_acquired,
            "lease_released": diag.instance_lease_released, "context_closed": diag.context_closed, "second": second}
out["P9_cancel_during_startup"] = asyncio.run(probe_cancel_during_start())

async def probe_cancel_running():
    j = []
    tmp = Path(tempfile.mkdtemp(prefix="iad4d_"))
    prof = make_profile(runtime=JournalRuntime(j), components=(HostedApplicationComponentRegistration(component=OrderedComponent("core", j), required=True),))
    eng, pub, _ = build_engine(definition_of(prof), tmp)
    t = asyncio.ensure_future(eng.run_until_stopped())
    await asyncio.sleep(0.5)
    t.cancel()
    await asyncio.gather(t, return_exceptions=True)
    diag = eng.diagnostics_snapshot()
    return {"journal": j, "state": eng.lifecycle_snapshot().state.value, "lease_released": diag.instance_lease_released,
            "context_closed": diag.context_closed}
out["P10_cancel_run_until_stopped"] = asyncio.run(probe_cancel_running())

async def probe_hanging_metric():
    tmp = Path(tempfile.mkdtemp(prefix="iad4e_"))
    j = []
    class Hang(RecordingPublisher):
        async def publish(self, event):
            if event.event_type is HostedApplicationEventType.LIFECYCLE_METRIC and event.payload.get("metric") == "drain_duration":
                await asyncio.sleep(3600)
            await super().publish(event)
    prof = make_profile(runtime=JournalRuntime(j))
    d = definition_of(prof)
    eng, pub, _ = build_engine(d, tmp, publisher=Hang())
    await eng.start()
    from intergrax.hosting.shutdown import compute_shutdown_budget_seconds
    pol = d.lifecycle_policy
    budget = compute_shutdown_budget_seconds(shutdown_policy=d.shutdown_policy, blocking_hook_timeout=pol.default_blocking_hook_timeout_seconds,
        observer_drain_timeout=pol.default_observer_hook_timeout_seconds, component_stop_budget=pol.default_blocking_hook_timeout_seconds,
        runtime_stop_budget=pol.default_blocking_hook_timeout_seconds, lease_release_timeout=pol.default_observer_hook_timeout_seconds,
        explicit_deadline_at=None, clock=eng.clock, requested_at=eng.clock.now())
    wait = min(budget + 5.0, 60.0)
    t0 = time.monotonic()
    try:
        await asyncio.wait_for(eng.stop(), wait)
        res = "stop_returned"
    except asyncio.TimeoutError:
        res = "stop_did_not_return"
    elapsed = round(time.monotonic() - t0, 1)
    return {"computed_shutdown_budget_s": budget, "waited_s": elapsed, "result": res}
out["P11_hanging_metric_exporter_in_stop"] = asyncio.run(probe_hanging_metric())
print(json.dumps(out, indent=1, default=str))
