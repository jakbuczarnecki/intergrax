"""IA-D4 mutation plugin: applies ONE in-memory mutant selected by IAD4_MUT (no file is edited)."""
import os

MUT = os.environ.get("IAD4_MUT", "")

def _apply():
    if MUT == "M1_forward_stop_order":
        import intergrax.hosting.engine.components as c
        orig = c.ComponentCoordinator._reverse_levels_for_ids
        def forward(self, ids):
            return tuple(reversed(orig(self, ids)))
        c.ComponentCoordinator._reverse_levels_for_ids = forward
    elif MUT == "M2_no_runtime_degradation":
        import intergrax.hosting.engine.components as c
        c.ComponentCoordinator._track_runtime_degradation = lambda self, resolved, health: None
    elif MUT == "M3_no_cancel":
        import intergrax.hosting.shutdown as s
        async def noop(self, budget, recorder, active_before):
            return None
        s.HostedApplicationShutdownExecutor._phase_cancel = noop
    elif MUT == "M4_no_exclusive_lock":
        import intergrax.hosting.instance.file_guard as g
        g.try_acquire_exclusive = lambda fd: None
    elif MUT == "M5_no_digest_verification":
        import intergrax.hosting.supervisor.supervisor as s
        s.HostedApplicationSupervisor._verify_engine_contract = lambda self, engine, launch: None
    elif MUT == "M6_no_metrics":
        import intergrax.hosting.engine.engine as e
        async def noop(self, metric, value, *, label=""):
            return None
        e.HostedApplicationEngine._publish_metric = noop
    elif MUT == "M7_never_restart":
        import intergrax.hosting.supervisor.restart as r
        r.HostedApplicationRestartPolicyEvaluator._should_restart = lambda self, exit_record: False
    elif MUT == "M8_public_surface_change":
        import intergrax.hosting.engine.engine as e
        e.HostedApplicationEngine.brand_new_public_method = lambda self: None
    elif MUT == "M9_drain_skipped":
        import intergrax.hosting.shutdown as s
        async def nodrain(self, budget, recorder, active_before):
            return True
        s.HostedApplicationShutdownExecutor._phase_drain = nodrain
    elif MUT == "M10_startup_rollback_skipped":
        import intergrax.hosting.engine.engine as e
        async def noclean(self, exc):
            self._reuse_blocked = True
        e.HostedApplicationEngine._startup_failure_cleanup = noclean
    elif MUT == "M11_lease_not_released":
        import intergrax.hosting.engine.engine as e
        async def norelease(self):
            return False
        e.HostedApplicationEngine._release_lease_verified = norelease
    elif MUT == "M12_exit_class_label_constant":
        import intergrax.hosting.engine.engine as e
        e.HostedApplicationEngine._exit_class_label = lambda self, recorder, *, failed, restart_requested=False: "clean_stop"

_apply()
