import asyncio, json, sqlite3, tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from intergrax.runtime.long_running.recurrence import RecurrenceMaterializer, RecurrenceRule
from intergrax.runtime.long_running.store import SQLiteTaskCheckpointStore
from tests.qualification.pdc_x.d7._support import CountingResumeExecutor, build_scheduler, paused_checkpoint, RESUME_TOKEN, TENANT_ID

T0 = datetime(2026, 3, 1, 10, 0, tzinfo=UTC)
async def main():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        ex = CountingResumeExecutor()
        sched, store, mat = build_scheduler(Path(d) / "l.db", ex)
        cp = paused_checkpoint(); store.save(cp)
        mat.start(RecurrenceRule(rule_id="rp", tenant_id=TENANT_ID, task_id=cp.task_id, resume_token=RESUME_TOKEN,
                                 interval_seconds=60, start_utc=T0.isoformat()))
        n1 = await sched.tick(now=T0 + timedelta(seconds=1))
        # a real resume re-pauses / advances the task: a newer checkpoint (new token) becomes latest
        cp2 = paused_checkpoint(); 
        cp2 = cp2.model_copy(update={"task_id": cp.task_id, "resume_token": "token-after-repause"})
        try:
            store.save(cp2, expected_revision=1)
            saved = True
        except Exception as e:
            saved = f"{type(e).__name__}: {e}"
        n2 = await sched.tick(now=T0 + timedelta(seconds=61))
        n3 = await sched.tick(now=T0 + timedelta(seconds=121))
        con = sqlite3.connect(Path(d) / "l.db")
        rows = con.execute("select schedule_id,status from scheduled_resumes order by run_at_utc").fetchall(); con.close()
        print(json.dumps({"fired": [n1, n2, n3], "saved_second_checkpoint": saved, "executor_calls": len(ex.schedule_ids), "rows": rows}, indent=1))
asyncio.run(main())
