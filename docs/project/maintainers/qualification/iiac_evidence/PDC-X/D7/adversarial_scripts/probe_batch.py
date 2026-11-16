import asyncio, json, sqlite3, tempfile, time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from intergrax.runtime.long_running.recurrence import RecurrenceMaterializer, RecurrenceRule
from intergrax.runtime.long_running.scheduler import LongRunningScheduler
from intergrax.runtime.long_running.store import SQLiteTaskCheckpointStore
from tests.qualification.pdc_x.d7._support import CountingResumeExecutor, paused_checkpoint, ok_task_result, RESUME_TOKEN, TENANT_ID

T0 = datetime(2026, 3, 1, 10, 0, tzinfo=UTC)

class Boom(CountingResumeExecutor):
    async def resume_task(self, task, *, checkpoint):
        self.schedule_ids.append(str(task.metadata.get("schedule_id")))
        if len(self.schedule_ids) == 1:
            raise RuntimeError("resume failed")
        return ok_task_result(checkpoint.task_id)

async def main():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        db = Path(d) / "l.db"
        store = SQLiteTaskCheckpointStore(db_path=db)
        ex = Boom()
        sched = LongRunningScheduler(store, ex, schedule_store=store, ledger=store, owner_id="s1",
                                     lease_seconds=1, recurrence=RecurrenceMaterializer(store))
        mat = RecurrenceMaterializer(store)
        for i in range(3):
            cp = paused_checkpoint(); store.save(cp)
            mat.start(RecurrenceRule(rule_id=f"rule-{i}", tenant_id=TENANT_ID, task_id=cp.task_id,
                resume_token=RESUME_TOKEN, interval_seconds=60, start_utc=T0.isoformat()))
        err = None
        try:
            await sched.tick(now=T0 + timedelta(seconds=5))
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
        time.sleep(1.5)
        n = await sched.tick(now=T0 + timedelta(seconds=10))
        con = sqlite3.connect(db)
        rows = con.execute("select schedule_id, status from scheduled_resumes order by schedule_id").fetchall(); con.close()
        print(json.dumps({"first_tick_error": err, "second_tick_fired": n, "executor_calls": ex.schedule_ids, "rows": rows}, indent=1))
asyncio.run(main())
