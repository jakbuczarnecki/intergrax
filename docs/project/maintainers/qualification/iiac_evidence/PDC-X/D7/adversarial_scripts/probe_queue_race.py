import json, threading, time
from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.queueing.contracts.task_queue import TaskRequest
from intergrax.queueing.providers.document_store import DocumentStoreTaskQueue

class SlowStore(InMemoryDocumentStore):
    def put(self, record):
        time.sleep(0.0005); return super().put(record)
st = SlowStore()
qa, qb = DocumentStoreTaskQueue(st), DocumentStoreTaskQueue(st)
for i in range(40):
    qa.enqueue(TaskRequest(tenant_id="t1", run_id=f"r{i}", task_name="n", payload=b"x"))
got = {"a": [], "b": []}
bar = threading.Barrier(2)
def run(name, qq):
    bar.wait()
    for _ in range(40):
        for hd, _ in qq.claim_pending(limit=4): got[name].append(hd.task_id)
ts = [threading.Thread(target=run, args=("a", qa)), threading.Thread(target=run, args=("b", qb))]
[t.start() for t in ts]; [t.join() for t in ts]
print(json.dumps({"a": len(got["a"]), "b": len(got["b"]), "double_claimed": len(set(got["a"]) & set(got["b"]))}))
