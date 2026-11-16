"""IA-D7 probes: recurrence + scheduler (read-only against product code)."""
import asyncio, tempfile, json, sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from intergrax.runtime.long_running.recurrence import (
    MisfirePolicy, RecurrenceMaterializer, RecurrenceRule, occurrence_schedule_id,
)
from intergrax.runtime.long_running.scheduled_resume import ScheduledResume
from intergrax.runtime.long_running.store import SQLiteTaskCheckpointStore
from tests.qualification.pdc_x.d7._support import (
    CountingResumeExecutor, build_scheduler, paused_checkpoint, RESUME_TOKEN, TENANT_ID,
)

T0 = datetime(2026, 3, 1, 10, 0, tzinfo=UTC)
out = {}

# ---- P1 tenant collision on identical rule_id
with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
    store = SQLiteTaskCheckpointStore(db_path=Path(d) / "l.db")
    m = RecurrenceMaterializer(store)
    def rule(tenant, task):
        return RecurrenceRule(rule_id="daily-report", tenant_id=tenant, task_id=task,
                              resume_token="tok", interval_seconds=3600, start_utc=T0.isoformat())
    a = m.start(rule("tenant-a", "task-a"))
    b = m.start(rule("tenant-b", "task-b"))
    rows = sqlite3.connect(Path(d) / "l.db").execute("select schedule_id, tenant_id from scheduled_resumes").fetchall()
    out["P1_tenant_collision"] = {"a_created": a is not None, "b_created": b is not None, "rows": rows}

# ---- P2 crash chain loss: claim, die (no complete, no materialize), lease expiry
async def p2():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        ex = CountingResumeExecutor()
        sched, store, mat = build_scheduler(Path(d) / "l.db", ex)
        cp = paused_checkpoint(); store.save(cp)
        rule = RecurrenceRule(rule_id="r-crash", tenant_id=TENANT_ID, task_id=cp.task_id,
            resume_token=RESUME_TOKEN, interval_seconds=60, start_utc=(T0).isoformat())
        mat.start(rule)
        claims = store.claim_due(before_utc_iso=(T0 + timedelta(seconds=5)).isoformat(), owner_id="dead", lease_seconds=0)
        await asyncio.sleep(0.05)
        # a survivor ticks much later
        fired = await sched.tick(now=T0 + timedelta(seconds=500))
        rows = sqlite3.connect(Path(d) / "l.db").execute("select schedule_id, status from scheduled_resumes").fetchall()
        return {"claimed": len(claims), "survivor_fired": fired, "rows": rows}
out["P2_crash_chain"] = asyncio.run(p2())

# ---- P2b materialization failure after completion: chain ends silently
async def p2b():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        ex = CountingResumeExecutor()
        sched, store, mat = build_scheduler(Path(d) / "l.db", ex)
        cp = paused_checkpoint(); store.save(cp)
        rule = RecurrenceRule(rule_id="r-fail", tenant_id=TENANT_ID, task_id=cp.task_id,
            resume_token=RESUME_TOKEN, interval_seconds=60, start_utc=T0.isoformat())
        mat.start(rule)
        orig = store.schedule
        calls = {"n": 0}
        def flaky(entry):
            calls["n"] += 1
            if calls["n"] == 2:
                raise sqlite3.OperationalError("database is locked")
            return orig(entry)
        store.schedule = flaky
        fired = await sched.tick(now=T0 + timedelta(seconds=1))
        fired2 = await sched.tick(now=T0 + timedelta(seconds=300))
        rows = sqlite3.connect(Path(d) / "l.db").execute("select schedule_id, status from scheduled_resumes").fetchall()
        return {"fired": fired, "fired_later": fired2, "rows": rows}
out["P2b_transient_materialize_failure"] = asyncio.run(p2b())

# ---- P3 non-UTC run_at_utc offsets
with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
    store = SQLiteTaskCheckpointStore(db_path=Path(d) / "l.db")
    now = datetime(2026, 12, 1, 8, 30, tzinfo=UTC)
    early = store.schedule(ScheduledResume(task_id="t", tenant_id="x", resume_token="k",
            run_at_utc="2026-12-01T05:00:00-05:00"))  # = 10:00Z, not due at 08:30Z
    late = store.schedule(ScheduledResume(task_id="t2", tenant_id="x", resume_token="k",
            run_at_utc="2026-12-01T10:00:00+02:00"))  # = 08:00Z, due at 08:30Z
    garbage = store.schedule(ScheduledResume(task_id="t3", tenant_id="x", resume_token="k", run_at_utc="not-a-time"))
    empty = store.schedule(ScheduledResume(task_id="t4", tenant_id="x", resume_token="k", run_at_utc=""))
    due = [e.task_id for e in store.list_due(before_utc_iso=now.isoformat())]
    out["P3_nonutc_run_at"] = {"due_at_0830Z": due, "expected_due": ["t2"]}

# ---- P4 unsatisfiable cron + overflow
r = RecurrenceRule(rule_id="never", tenant_id="x", task_id="t", resume_token="k",
                   cron="0 0 30 2 *", start_utc=T0.isoformat())
import time; t=time.time(); fo = r.first_occurrence()
out["P4_unsatisfiable_cron"] = {"accepted": True, "first_occurrence": str(fo), "scan_seconds": round(time.time()-t,2)}
try:
    big = RecurrenceRule(rule_id="big", tenant_id="x", task_id="t", resume_token="k",
                         interval_seconds=10**15, start_utc=T0.isoformat())
    out["P4_overflow_rule_accepted"] = True
    try:
        out["P4_overflow_next"] = str(big.next_after(T0))
    except Exception as e:
        out["P4_overflow_next"] = f"{type(e).__name__}: {e}"
except Exception as e:
    out["P4_overflow_rule_accepted"] = f"{type(e).__name__}"

# ---- P7 DST gap semantics
ny = RecurrenceRule(rule_id="gap", tenant_id="x", task_id="t", resume_token="k",
        cron="30 2 * * *", timezone="America/New_York", start_utc=datetime(2026,3,7,0,0,tzinfo=UTC).isoformat())
occ = ny.occurrences_between(datetime(2026,3,6,tzinfo=UTC), datetime(2026,3,10,tzinfo=UTC), 10)
out["P7_gap_30_2"] = [str(o.astimezone(ZoneInfo("America/New_York"))) for o in occ]
ny0 = RecurrenceRule(rule_id="gap0", tenant_id="x", task_id="t", resume_token="k",
        cron="0 2 * * *", timezone="America/New_York", start_utc=datetime(2026,3,7,0,0,tzinfo=UTC).isoformat())
occ = ny0.occurrences_between(datetime(2026,3,6,tzinfo=UTC), datetime(2026,3,10,tzinfo=UTC), 10)
out["P7_gap_0_2"] = [str(o.astimezone(ZoneInfo("America/New_York"))) for o in occ]
# two cron entries mapping into same instant: '30 2,3'
ny2 = RecurrenceRule(rule_id="gap2", tenant_id="x", task_id="t", resume_token="k",
        cron="30 2,3 * * *", timezone="America/New_York", start_utc=datetime(2026,3,7,0,0,tzinfo=UTC).isoformat())
occ = ny2.occurrences_between(datetime(2026,3,6,tzinfo=UTC), datetime(2026,3,10,tzinfo=UTC), 20)
out["P7_gap_30_2_3"] = [str(o.astimezone(ZoneInfo("America/New_York"))) for o in occ]
# fall-back fixed hour
fb = RecurrenceRule(rule_id="fb", tenant_id="x", task_id="t", resume_token="k",
        cron="30 1 * * *", timezone="America/New_York", start_utc=datetime(2026,10,31,0,0,tzinfo=UTC).isoformat())
occ = fb.occurrences_between(datetime(2026,10,31,tzinfo=UTC), datetime(2026,11,3,tzinfo=UTC), 10)
out["P7_fallback_30_1"] = [str(o) for o in occ]
# half-hour zone & Lord Howe (30-min DST)
lh = RecurrenceRule(rule_id="lh", tenant_id="x", task_id="t", resume_token="k",
        cron="15 2 * * *", timezone="Australia/Lord_Howe", start_utc=datetime(2026,10,3,0,0,tzinfo=UTC).isoformat())
occ = lh.occurrences_between(datetime(2026,10,3,tzinfo=UTC), datetime(2026,10,7,tzinfo=UTC), 10)
out["P7_lord_howe"] = [str(o.astimezone(ZoneInfo("Australia/Lord_Howe"))) for o in occ]

print(json.dumps(out, indent=1, default=str))
