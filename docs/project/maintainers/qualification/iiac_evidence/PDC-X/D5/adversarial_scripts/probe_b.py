"""IA-D5 adversarial probes B (registry, tenant, wiring, human queue)."""
from __future__ import annotations

import sys
import tempfile
import threading
import time
from pathlib import Path

sys.path.insert(0, "D:/Projekty/intergrax/.claude/worktrees/pdc-x-ia-d5-fb28f3")

from intergrax.runtime.architecture.online_evaluation_registry import (  # noqa: E402
    FileOnlineEvaluationRegistry, TenantScopedOnlineEvaluationRegistry, OnlineEvaluationTenantBoundaryError)
from intergrax.runtime.architecture.user_satisfaction import (  # noqa: E402
    UserSatisfactionEvent, SatisfactionSignal, record_user_satisfaction)
from intergrax.runtime.architecture.online_evaluation import record_shadow_observation  # noqa: E402
from intergrax.runtime.evaluation.human_review_sample_queue import HumanReviewSampleQueue  # noqa: E402


def p(name, val):
    print(f"[{name}] {val}")


tmp = Path(tempfile.mkdtemp(prefix="iad5_"))
reg = FileOnlineEvaluationRegistry(tmp / "obs.json")

# Q1: production writers do not stamp tenant
ev = UserSatisfactionEvent(tenant_id="tenant-a", task_id="t", run_id="r-a", signal=SatisfactionSignal.THUMBS_UP, score=9.0)
record_user_satisfaction(ev, registry=reg)
record_shadow_observation(run_id="r-b", agent_id="ag", scenario_id="sc", passed=True, score=1.0, registry=reg)
all_obs = reg.list_observations()
p("Q1 tenant_id of user-satisfaction obs written for tenant-a", getattr(all_obs[0], "tenant_id", "MISSING"))
p("Q1 tenant-b scoped view sees tenant-a's satisfaction obs", len(TenantScopedOnlineEvaluationRegistry(reg, "tenant-b").list_observations()))
p("Q1 legacy list_observations() returns rows to any caller", len(all_obs))

# Q2: tenant-scoped append then legacy list leaks across tenants
TenantScopedOnlineEvaluationRegistry(reg, "tenant-a").append(all_obs[1])
p("Q2 legacy unscoped list sees tenant-a stamped row", [getattr(o, "tenant_id", None) for o in reg.list_observations()])

# Q3: existence oracle
scoped_b = TenantScopedOnlineEvaluationRegistry(reg, "tenant-b")
try:
    scoped_b.get_observation(all_obs[1].observation_id)
except Exception as exc:
    p("Q3 tenant-b get of tenant-a id", type(exc).__name__)
try:
    scoped_b.get_observation("does-not-exist")
except Exception as exc:
    p("Q3 tenant-b get of unknown id", type(exc).__name__)

# Q4: observation_id collision between tenants denies owner's valid read
reg2 = FileOnlineEvaluationRegistry(tmp / "obs2.json")
o = record_shadow_observation(run_id="r1", agent_id="a", scenario_id="s", passed=True, score=1.0, registry=TenantScopedOnlineEvaluationRegistry(reg2, "tenant-a"))
record_shadow_observation(run_id="r1", agent_id="a", scenario_id="s", passed=False, score=0.0, registry=TenantScopedOnlineEvaluationRegistry(reg2, "tenant-b"))
try:
    got = TenantScopedOnlineEvaluationRegistry(reg2, "tenant-b").get_observation(o.observation_id)
    p("Q4 tenant-b get same id", got.passed)
except Exception as exc:
    p("Q4 tenant-b get same id (own row exists) ->", type(exc).__name__)
p("Q4 tenant-b list own", len(TenantScopedOnlineEvaluationRegistry(reg2, "tenant-b").list_observations()))

# Q5: time-based lock lease: holder > stale_lock_seconds loses exclusivity
path = tmp / "obs3.json"
holder = FileOnlineEvaluationRegistry(path, stale_lock_seconds=0.3, lock_timeout_seconds=5)
intruder = FileOnlineEvaluationRegistry(path, stale_lock_seconds=0.3, lock_timeout_seconds=5)
events: list[str] = []

def hold():
    with holder._exclusive():
        events.append("A-in")
        time.sleep(1.0)
        events.append("A-out")

def intrude():
    time.sleep(0.1)
    with intruder._exclusive():
        events.append("B-in")
        time.sleep(1.5)
        events.append("B-out")

t1 = threading.Thread(target=hold); t2 = threading.Thread(target=intrude)
t1.start(); t2.start(); t1.join(); t2.join()
p("Q5 critical-section order (overlap if B-in before A-out)", events)

# Q6: lock file left by A's exit unlinks B's lock -> C can enter while B inside
path = tmp / "obs4.json"
a = FileOnlineEvaluationRegistry(path, stale_lock_seconds=0.3, lock_timeout_seconds=5)
b = FileOnlineEvaluationRegistry(path, stale_lock_seconds=0.3, lock_timeout_seconds=5)
c = FileOnlineEvaluationRegistry(path, stale_lock_seconds=60, lock_timeout_seconds=5)
ev2: list[str] = []
def ta():
    with a._exclusive():
        ev2.append("A-in"); time.sleep(0.8); ev2.append("A-out")
def tb():
    time.sleep(0.1)
    with b._exclusive():
        ev2.append("B-in"); time.sleep(1.6); ev2.append("B-out")
def tc():
    time.sleep(1.0)
    with c._exclusive():
        ev2.append("C-in"); time.sleep(0.1); ev2.append("C-out")
ths = [threading.Thread(target=f) for f in (ta, tb, tc)]
[t.start() for t in ths]; [t.join() for t in ths]
p("Q6 A-out unlinks B's lock; C enters while B inside", ev2)

# Q7: human review queue
q = HumanReviewSampleQueue()
s = q.enqueue(run_id="r", agent_id="a", scenario_id="s", reason="x")
p("Q7 mark_reviewed with empty reviewer id", q.mark_reviewed(s.sample_id, reviewer_id="").reviewer_id == "")
p("Q7 re-review by other id allowed", q.mark_reviewed(s.sample_id, reviewer_id="anyone").reviewer_id)
p("Q7 queue has tenant field", "tenant_id" in s.model_fields)
