# HARDENING-9 — NPSC-5F Event Delivery QoS Protected Drift Re-freeze

**Task:** `HARDENING_9_NPSC5F_EVENT_DELIVERY_QOS_PROTECTED_DRIFT_REFREEZE`  
**Status:** `QUALIFIED / RE-FROZEN / PASS`  
**START_HEAD:** `1c21eb774e7dd6c24148cf67be98ddc894cbd3c9`  
**QUALIFICATION_HEAD:** `1c21eb774e7dd6c24148cf67be98ddc894cbd3c9`  
**END_REMOTE_HEAD:** `1c21eb774e7dd6c24148cf67be98ddc894cbd3c9`  
**OLD_FINAL_BASELINE:** `33576b80521dda7dfc0e5895c943f91dfebffa94`  
**NEW_FINAL_BASELINE (`FINAL_QUALIFIED_HEAD`):** `1c21eb774e7dd6c24148cf67be98ddc894cbd3c9`

## Executive summary

Integrated `development` gained **OBS-DELIVERY-QOS-SCALE** (`97503fd39`) and **OBS-DELIVERY-QOS-SCALE-R1** (`b84eb1c2b`): process-local bounded event delivery admission partitioning via `EventDeliveryAdmissionPolicyPort`, `critical_reserved_capacity` on `EventDeliveryPolicy`, and atomic non-critical slot accounting in `BoundedEventSink`. Durable canonical execution evidence (persistence, journal, export governance) is unchanged; drift is delivery-buffer QoS only.

**Production code in this re-freeze commit:** NO — sentinel baseline + qualification record + test SHA pins only.

## Drift commits (`33576b80..1c21eb774`)

| Commit | Protected files | Classification |
|--------|-----------------|----------------|
| `97503fd3929d6cb14f234626ebddb38b64e51786` | `bounded_event_sink.py`, `enterprise_default_event_delivery_admission_policy.py` (new) | **A** NON_BREAKING_ENTERPRISE_EVOLUTION |
| `97503fd3929d6cb14f234626ebddb38b64e51786` | `event_delivery.py`, `runtime_event_delivery_wiring.py`, `sub_profiles.py` | **A** (contract + composition; outside Final BREAKING classifier except wiring = QUALIFIED_COMPATIBLE prefix) |
| `b84eb1c2b385271a3b2c03ade7845949967fee7a` | `bounded_event_sink.py` | **B** BEHAVIORAL_BUT_QUALIFIED_CHANGE (atomic admission / TOCTOU close) |
| `1c21eb774…` … `8d0299da2` | — | OUTSIDE protected production surface |

## Protected production files (Final classifier)

| File | Change type | Qualified? |
|------|-------------|------------|
| `intergrax/runtime/observability/event_delivery/bounded_event_sink.py` | Admission accounting + injectable policy | YES |
| `intergrax/runtime/observability/event_delivery/enterprise_default_event_delivery_admission_policy.py` | Default admission limits | YES |

## Semantic qualification

| Question | Answer |
|----------|--------|
| WHAT | Reserved CRITICAL buffer slots; non-critical cap via port; `_non_critical_buffered` under lock |
| WHY | Prevent BEST_EFFORT/IMPORTANT from exhausting bounded sink before CRITICAL admission |
| WHO | `intergrax/contracts/event_delivery.py` owns contracts; runtime observability implements |
| Contract | `EventDeliveryAdmissionPolicyPort`, `EventDeliveryPolicy.critical_reserved_capacity` |
| Evidence semantics altered? | **NO** — contract doc: transport-only; durable evidence not owned by delivery layer |
| Delivery only? | **YES** |
| Ordering | Single `queue.Queue(maxsize=max_capacity)`; FIFO; priority affects admission only |

## Architecture assessment

| Check | Result |
|-------|--------|
| Contract-first | YES |
| Admission policy replaceable | YES (`admission_policy=` DI; `test_obs_delivery_qos_scale` custom port proof) |
| Default implementation only default | YES (`EnterpriseDefaultEventDeliveryAdmissionPolicy` used when caller omits policy) |
| Single capacity authority | YES (`max_capacity` on policy; one physical queue) |
| Duplicate admission authority | NO (single `enterprise_default_critical_reserved_slots` in wiring + policy module) |
| Layer boundaries preserved | YES |
| Vendor neutrality preserved | YES |
| Private coupling | NO |
| Reflection workaround | NO |

## QoS semantics (@ qualified commits)

| Priority | Semantics verified |
|----------|-------------------|
| BEST_EFFORT | Non-blocking; DROPPED when partition/full |
| IMPORTANT | Bounded wait → DEFERRED; uses non-critical partition |
| CRITICAL | `put_nowait` on full queue → REJECTED; may use reserved slots; COMPLETION obligation |
| Reserved capacity | `0 <= reserve <= max_capacity`; non-critical limit = `max_capacity - reserve` |
| Shutdown | Sentinel + drain; quota condition notify on close |

## Concurrency

- `_reserve_non_critical_slot` holds `_quota_condition` lock through counter increment; enqueue failure releases slot in `finally`.
- `_dequeue_accounting` decrements on non-CRITICAL dequeue.
- R1 atomic admission tests: `test_obs_delivery_qos_scale_r1_atomic_admission.py` + extended QoS scale module.

## Gates (session)

| Gate | Before re-freeze | After re-freeze |
|------|------------------|-----------------|
| Final protected drift | FAIL (2 BREAKING paths) | PASS |
| NPSC5F mandatory matrix | FAIL (drift) | PASS |
| R1 / R2 / R3 drift | PASS | PASS |
| H1 upstream reconciliation | PASS | PASS |
| QoS tests | PASS | PASS |

## Freeze state

| Sentinel | Action |
|----------|--------|
| `R1_POST_R2_QUALIFIED_BASELINE_SHA` | UNCHANGED |
| `R2_POST_QUALIFIED_BASELINE_SHA` | UNCHANGED |
| `R3_POST_QUALIFIED_BASELINE_SHA` | UNCHANGED |
| `NPSC5F_FINAL_EVIDENCE_PLANE_BASELINE_SHA` | `33576b80` → `1c21eb774` |
| `NPSC5F_R3_H1_QUALIFIED_BASELINE_SHA` | UNCHANGED |
| `NPSC5F_R3_H1_QUALIFICATION_RECORD_SHA` | UNCHANGED |

## Head movement

No new commits on protected surfaces between session open and re-freeze pin (`END_REMOTE_HEAD == START_HEAD`).

## GitHub audit note

> Wprowadzone zmiany muszą zostać zaudytowane na podstawie kodu z GitHub przed uznaniem zadania za ostatecznie zamknięte.
