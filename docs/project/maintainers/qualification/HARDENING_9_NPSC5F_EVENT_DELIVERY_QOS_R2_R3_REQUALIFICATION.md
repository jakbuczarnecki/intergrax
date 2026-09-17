# HARDENING-9 — NPSC-5F Event Delivery QoS R2+R3 Requalification

**Task:** `HARDENING_9_NPSC5F_EVENT_DELIVERY_QOS_R2_R3_REQUALIFICATION`  
**Status:** `QUALIFIED / RE-FROZEN / PASS` (pending GitHub audit)

| Field | SHA |
| --- | --- |
| `LAST_INTEGRATED_H9_QUALIFICATION_COMMIT` | `242db6f00634a866e3bb30807ff8ae6db3f36e2d` |
| `START_HEAD` | `eba9f6bff1e19c14e9590ff22fb6312e27463450` |
| `FINAL_QUALIFIED_HEAD` | `eba9f6bff1e19c14e9590ff22fb6312e27463450` |
| `PRE_REFREEZE_REMOTE_HEAD` | `eba9f6bff1e19c14e9590ff22fb6312e27463450` |
| Old Final baseline | `1c21eb774e7dd6c24148cf67be98ddc894cbd3c9` |
| New Final baseline | `eba9f6bff1e19c14e9590ff22fb6312e27463450` |

## R2 / R3 commits

| Commit | Summary |
| --- | --- |
| `f6b4b542b` | OBS-DELIVERY-QOS-SCALE-R2: linearize admission with shutdown |
| `eba9f6bff` | OBS-DELIVERY-QOS-SCALE-R3: fail closed on pending enqueue shutdown timeout |

## H9-relevant delta (`242db6f0..eba9f6bff`)

| Commit | Files (H9 surface) | Classification |
| --- | --- | --- |
| `f6b4b542b` | `bounded_event_sink.py`, `test_obs_delivery_qos_scale.py`, `OBSERVABILITY.md` | **B** BEHAVIORAL_BUT_QUALIFIED_CHANGE |
| `eba9f6bff` | `bounded_event_sink.py`, `test_obs_delivery_qos_scale.py`, `OBSERVABILITY.md` | **B** BEHAVIORAL_BUT_QUALIFIED_CHANGE |
| Other commits in range | governance, multiplayer, scheduling, CE-01, docs | **UNRELATED** (outside Final BREAKING classifier) |

Pre re-freeze Final BREAKING drift vs old baseline: **only** `intergrax/runtime/observability/event_delivery/bounded_event_sink.py`.

## R2 assessment

- **Admission/shutdown linearization:** `_begin_physical_enqueue` / `_pending_physical_enqueue` under `_quota_condition`; `close()` sets `_stop` and `notify_all` before drain.
- **Pending physical enqueue:** reserved on admission paths; released in `finally` via `_finish_physical_enqueue`.
- **Sentinel ordering:** `_wait_pending_physical_enqueue_drain` completes before `_enqueue_shutdown_sentinel`.
- **Lifecycle race:** quota waiters observe `_stop` and reject; tests `test_r2_*`.
- **Verdict:** **PASS**

## R3 assessment

- **Shutdown timeout:** `_wait_pending_physical_enqueue_drain` uses `time.monotonic()` + `drain_shutdown_timeout_seconds`; on expiry → `mark_unhealthy` + `EventDeliveryBoundaryError(INTERNAL_ERROR)`; **no sentinel**, **no downstream.close**.
- **Worker-dead:** `worker_dead_before_shutdown` and in-drain worker check → `SINK_UNAVAILABLE`.
- **Second close / post-failure publish:** `_raise_if_shutdown_not_successful`; publish returns REJECTED when unhealthy/stopped.
- **Verdict:** **PASS**

## Architecture

Contract-first, pluginability (`EventDeliveryAdmissionPolicyPort` injectable), single `queue.Queue(maxsize=max_capacity)`, layer boundaries and vendor neutrality preserved. No reflection workaround; `_pending_physical_enqueue` remains implementation-only.

## Concurrency

Lock order: `_quota_condition` guards stop, counters, and drain wait (predicate loop). No confirmed deadlock. Lost/spurious wakeup handled via `while` + `notify_all` on close and counter release.

## Freeze safety

| Sentinel | Action |
| --- | --- |
| `R1_POST_R2_QUALIFIED_BASELINE_SHA` | UNCHANGED |
| `R2_POST_QUALIFIED_BASELINE_SHA` | UNCHANGED |
| `R3_POST_QUALIFIED_BASELINE_SHA` | UNCHANGED |
| `NPSC5F_R3_H1_*` | UNCHANGED |
| `NPSC5F_FINAL_EVIDENCE_PLANE_BASELINE_SHA` | `1c21eb774` → `eba9f6bff` |

## Test evidence (clean worktree @ `eba9f6bff`)

QoS module 29/29 PASS; R2/R3 white-box tests in `test_obs_delivery_qos_scale.py`; event-delivery conformance cluster PASS; pre re-freeze Final drift FAIL (known single path); post re-freeze gates per qualification commit.

## Superseded authority

Local commit `c32c5470…` and baseline `a4aa388f…` are **not** used (pre-R3, not GitHub-auditable).
