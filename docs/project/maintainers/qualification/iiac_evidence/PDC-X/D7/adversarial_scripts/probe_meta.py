import asyncio, json, sqlite3, tempfile, time, sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from intergrax.runtime.long_running.recurrence import MisfirePolicy, RecurrenceMaterializer, RecurrenceRule
from intergrax.runtime.long_running.scheduler import LongRunningScheduler
from intergrax.runtime.long_running.store import SQLiteTaskCheckpointStore
from tests.qualification.pdc_x.d7._support import CountingResumeExecutor, paused_checkpoint, RESUME_TOKEN, TENANT_ID

T0 = datetime(2026, 3, 1, 10, 0, tzinfo=UTC)
async def run(late_seconds):
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        db = Path(d) / "l.db"
        store = SQLiteTaskCheckpointStore(db_path=db)
        ex = CountingResumeExecutor()
        sched = LongRunningScheduler(store, ex, schedule_store=store, ledger=store, owner_id="s1",
                                     recurrence=RecurrenceMaterializer(store))
        cp = paused_checkpoint(); store.save(cp)
        rule = RecurrenceRule(rule_id="inj", tenant_id=TENANT_ID, task_id=cp.task_id, resume_token=RESUME_TOKEN,
                              interval_seconds=1, start_utc=T0.isoformat(), misfire=MisfirePolicy.catch_up(5000))
        # recurrence smuggled through plain one-shot resume_metadata on the public scheduler API
        sched.schedule_resume(task_id=cp.task_id, tenant_id=TENANT_ID, resume_token=RESUME_TOKEN,
                              run_at_utc=T0.isoformat(), resume_metadata={"recurrence_rule": rule.model_dump(mode="json")})
        t = time.time()
        fired = 0
        for _ in range(400):
            n = await sched.tick(now=T0 + timedelta(seconds=late_seconds))
            if n == 0: break
            fired += n
        con = sqlite3.connect(db); rows = con.execute("select count(*) from scheduled_resumes").fetchone()[0]; con.close()
        return {"late_seconds": late_seconds, "fired": fired, "rows": rows, "seconds": round(time.time() - t, 2)}
for late in (50, 100, 200):
    print(json.dumps(asyncio.run(run(late))), flush=True)
