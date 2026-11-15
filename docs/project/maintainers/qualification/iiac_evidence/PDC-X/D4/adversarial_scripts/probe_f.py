"""IA-D4 probe F: hung runtime.ready() after start -> is readiness still reported as ready (stale)?"""
import asyncio, json, sys, tempfile
from pathlib import Path
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from tests.qualification.pdc_x.d4._support import JournalRuntime, build_engine, definition_of, make_profile
class HangingReady(JournalRuntime):
    hang = False
    async def ready(self, context):
        if HangingReady.hang:
            await asyncio.sleep(3600)
        return self.started
async def main():
    tmp = Path(tempfile.mkdtemp(prefix="iad4i_"))
    rt = HangingReady()
    eng, pub, _ = build_engine(definition_of(make_profile(runtime=rt)), tmp)
    await eng.start()
    h0 = eng.health_snapshot()
    HangingReady.hang = True
    await asyncio.sleep(1.5)  # poll interval 0.05s: >25 polls would have run
    h1 = eng.health_snapshot()
    rt.started = False  # runtime really becomes not-ready, but ready() hangs
    await asyncio.sleep(1.0)
    h2 = eng.health_snapshot()
    res = {"ready_before": h0.ready, "ready_while_ready_hangs": h1.ready, "ready_after_runtime_died_but_ready_hangs": h2.ready,
           "health_evaluation_failed": h2.health_evaluation_failed, "last_evaluated_at_changed": h2.last_evaluated_at != h0.last_evaluated_at}
    HangingReady.hang = False
    try:
        await asyncio.wait_for(eng.stop(), 60)
        res["stop"] = "returned"
    except asyncio.TimeoutError:
        res["stop"] = "did_not_return_in_60s"
    return res
print(json.dumps(asyncio.run(main()), indent=1))
