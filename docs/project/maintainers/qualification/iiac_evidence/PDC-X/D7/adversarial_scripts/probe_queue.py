"""IA-D7 probes: queue contract / DocumentStore provider (read-only vs product)."""
import json, threading
from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.queueing.contracts.task_queue import (
    DeadLetterReason, TaskRequest, TaskStatus, TaskExecutionIdentity,
)
from intergrax.queueing.providers.document_store import DocumentStoreTaskQueue
from intergrax.queueing.worker.retry_policy import RetryPolicy

out = {}
def pol(n, backoff=0.0, mx=None):
    return RetryPolicy(max_retries=n, initial_backoff_seconds=backoff, backoff_multiplier=2.0,
                       max_backoff_seconds=mx, jitter=False, retry_on_lock_conflict=True,
                       retry_on_handler_exception=True)

# P5: terminal state is not sticky
st = InMemoryDocumentStore(); q = DocumentStoreTaskQueue(st)
h = q.enqueue(TaskRequest(tenant_id="t1", run_id="r", task_name="n", payload=b"x"))
q.claim_pending(); q.mark_dead_lettered(h, reason=DeadLetterReason.RETRY_BUDGET_EXHAUSTED, error_message="boom")
before = q.get_result(h)
q.mark_succeeded(h, output=b"late")            # stale worker completes after dead-letter
after = q.get_result(h)
raw = st.get("intergrax.task_queue.v1:t1", h.task_id).data
out["P5_terminal_overwrite"] = {"before": before.status.value, "after": after.status.value,
                                "stale_dead_letter_field_remains": raw.get("dead_letter") is not None,
                                "result_dead_letter_attr": after.dead_letter}
q.mark_failed(h, error_message="again"); out["P5_after_failed"] = q.get_status(h).value

# P5b: retry_or_dead_letter on an already terminal task re-opens it
st = InMemoryDocumentStore(); q = DocumentStoreTaskQueue(st)
h = q.enqueue(TaskRequest(tenant_id="t1", run_id="r", task_name="n", payload=b"x"))
q.claim_pending(); q.mark_succeeded(h, output=b"ok")
res = q.retry_or_dead_letter(h, error_message="late failure", retry_policy=pol(3))
out["P5b_retry_on_succeeded"] = {"returned": res.value, "now": q.get_status(h).value}

# P6 retry window with max_backoff None
try:
    out["P6_window"] = pol(3).max_retry_window_seconds()
except Exception as e:
    out["P6_window"] = f"{type(e).__name__}: {e}"
try:
    out["P6_window_cap"] = pol(3, 1.0, 10.0).max_retry_window_seconds()
except Exception as e:
    out["P6_window_cap"] = f"{type(e).__name__}: {e}"

# P9 head-of-line blocking by back-off rows (index scan window = max(limit*4, 32))
st = InMemoryDocumentStore(); q = DocumentStoreTaskQueue(st)
hs = [q.enqueue(TaskRequest(tenant_id="t1", run_id=f"r{i}", task_name="n", payload=b"x")) for i in range(40)]
claimed = q.claim_pending(limit=40)
for hd, _ in claimed:
    q.retry_or_dead_letter(hd, error_message="x", retry_policy=pol(5, 3600.0, 7200.0))
fresh = q.enqueue(TaskRequest(tenant_id="t1", run_id="fresh", task_name="n", payload=b"x"))
got = q.claim_pending(limit=4)
out["P9_hol"] = {"backed_off": 40, "fresh_claimed": [g[0].task_id == fresh.task_id for g in got], "claimed_n": len(got)}

# P10 idempotency key after dead-letter
st = InMemoryDocumentStore(); q = DocumentStoreTaskQueue(st)
r = TaskRequest(tenant_id="t1", run_id="r", task_name="n", payload=b"x", idempotency_key="k1")
h1 = q.enqueue(r); q.claim_pending(); q.mark_dead_lettered(h1, reason=DeadLetterReason.RETRY_BUDGET_EXHAUSTED, error_message="x")
h2 = q.enqueue(r)
out["P10_idem_after_dlq"] = {"same_handle": h1.task_id == h2.task_id, "status": q.get_status(h2).value}

# P11 two queue instances over one store: no cross-instance claim exclusion
st = InMemoryDocumentStore()
qa, qb = DocumentStoreTaskQueue(st), DocumentStoreTaskQueue(st)
for i in range(50):
    qa.enqueue(TaskRequest(tenant_id="t1", run_id=f"r{i}", task_name="n", payload=b"x"))
got = {"a": [], "b": []}
def run(name, qq):
    for _ in range(30):
        for hd, _ in qq.claim_pending(limit=8): got[name].append(hd.task_id)
ta = threading.Thread(target=run, args=("a", qa)); tb = threading.Thread(target=run, args=("b", qb))
ta.start(); tb.start(); ta.join(); tb.join()
dups = set(got["a"]) & set(got["b"])
out["P11_two_instances"] = {"a": len(got["a"]), "b": len(got["b"]), "double_claimed": len(dups)}

# P12 tenant isolation of handle
st = InMemoryDocumentStore(); q = DocumentStoreTaskQueue(st)
h = q.enqueue(TaskRequest(tenant_id="ta", run_id="r", task_name="n", payload=b"secret", idempotency_key="k"))
from intergrax.queueing.contracts.task_queue import TaskHandle
foreign = TaskHandle(task_id=h.task_id, provider=h.provider, tenant_id="tb")
out["P12_tenant_isolation"] = {"foreign_status": q.get_status(foreign).value, "foreign_result": q.get_result(foreign),
                               "foreign_cancel": q.cancel(foreign), "foreign_list": [s.task_id for s in q.list_tasks("tb")]}
h2 = q.enqueue(TaskRequest(tenant_id="tb", run_id="r", task_name="n", payload=b"y", idempotency_key="k"))
out["P12_idem_cross_tenant_distinct"] = h2.task_id != h.task_id

# P13 purge / list
out["P13_purge_completed_default"] = q.purge_completed("ta")
print(json.dumps(out, indent=1, default=str))
