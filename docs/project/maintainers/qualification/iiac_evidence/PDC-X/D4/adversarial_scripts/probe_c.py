"""IA-D4 adversarial probes C: classifier/metric parity, supervisor-level conflict, restart storm."""
import asyncio, json, sys, tempfile
from pathlib import Path
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
out = {}
from intergrax.hosting.engine.diagnostics import HOSTED_APPLICATION_STARTUP_FAILURE_PHASES, HostedApplicationFailurePhase
import intergrax.hosting.supervisor.classification as cl
import inspect
src = inspect.getsource(cl.HostedApplicationExitClassifier.classify_terminal_result)
start = src.index("startup_phases = {")
block = src[start:src.index("}", start)]
local = {p for p in HostedApplicationFailurePhase if f"HostedApplicationFailurePhase.{p.name}" in block}
out["Q1_startup_phase_sets_equal"] = local == set(HOSTED_APPLICATION_STARTUP_FAILURE_PHASES)
out["Q1_only_in_classifier"] = sorted(p.name for p in local - set(HOSTED_APPLICATION_STARTUP_FAILURE_PHASES))
out["Q1_only_in_diagnostics"] = sorted(p.name for p in set(HOSTED_APPLICATION_STARTUP_FAILURE_PHASES) - local)

from intergrax.hosting import (HostedApplicationEventType, HostedApplicationSupervisor, RestartPolicy,
                               HostedApplicationComponentRegistration)
from intergrax.hosting.contracts.policies import RestartMode
from intergrax.hosting.control import HostedApplicationControlCoordinator
from intergrax.hosting.metrics import HostedApplicationLifecycleMetric
from tests.qualification.pdc_x.d4._support import (JournalRuntime, RecordingPublisher, SystemClock, build_engine,
                                                   definition_of, make_profile, OrderedComponent)

async def conflict_case():
    tmp = Path(tempfile.mkdtemp(prefix="iad4f_"))
    prof1 = make_profile(runtime=JournalRuntime())
    holder, _, _ = build_engine(definition_of(prof1), tmp, instance_id="holder-1")
    await holder.start()
    try:
        prof2 = make_profile(runtime=JournalRuntime(), restart=RestartPolicy(mode=RestartMode.ALWAYS, max_attempts=2,
                             initial_backoff_seconds=0.05, max_backoff_seconds=0.1, jitter_ratio=0.0))
        d = definition_of(prof2)
        clock = SystemClock(); control = HostedApplicationControlCoordinator(clock=clock); pub = RecordingPublisher()
        n = iter(range(10))
        def ef(launch):
            e, _, _ = build_engine(launch.definition, tmp, publisher=pub, control=launch.control, instance_id=launch.instance_id)
            return e
        sup = HostedApplicationSupervisor(definition=d, engine_factory=ef, control=control, event_publisher=pub, clock=clock,
                                          instance_id_generator=lambda: f"conf-{next(n)}")
        res = await asyncio.wait_for(sup.run(), 20)
        labels = [e.payload["label"] for e in pub.of_type(HostedApplicationEventType.LIFECYCLE_METRIC) if e.payload["metric"] == "exit_class"]
        return {"final": res.final_exit.exit_kind.value, "attempts": len(res.attempts), "exit_class_metric_labels": labels,
                "exhausted": res.restart_exhausted}
    finally:
        await holder.stop()
out["Q2_supervisor_instance_conflict_with_ALWAYS_restart"] = asyncio.run(conflict_case())

async def clean_exit_always():
    # RestartMode.ALWAYS with a clean stop requested by control must not loop
    tmp = Path(tempfile.mkdtemp(prefix="iad4g_"))
    prof = make_profile(runtime=JournalRuntime(), restart=RestartPolicy(mode=RestartMode.ALWAYS, max_attempts=3,
                         initial_backoff_seconds=0.05, max_backoff_seconds=0.1, jitter_ratio=0.0))
    d = definition_of(prof)
    clock = SystemClock(); control = HostedApplicationControlCoordinator(clock=clock); pub = RecordingPublisher()
    n = iter(range(10))
    def ef(launch):
        e, _, _ = build_engine(launch.definition, tmp, publisher=pub, control=launch.control, instance_id=launch.instance_id)
        return e
    sup = HostedApplicationSupervisor(definition=d, engine_factory=ef, control=control, event_publisher=pub, clock=clock,
                                      instance_id_generator=lambda: f"al-{next(n)}")
    t = asyncio.ensure_future(sup.run())
    for _ in range(100):
        if pub.of_type(HostedApplicationEventType.APPLICATION_READY): break
        await asyncio.sleep(0.05)
    control.request_shutdown("probe.stop")
    res = await asyncio.wait_for(t, 20)
    return {"attempts": len(res.attempts), "final": res.final_exit.exit_kind.value}
out["Q3_always_restart_honours_explicit_shutdown"] = asyncio.run(clean_exit_always())
print(json.dumps(out, indent=1, default=str))
