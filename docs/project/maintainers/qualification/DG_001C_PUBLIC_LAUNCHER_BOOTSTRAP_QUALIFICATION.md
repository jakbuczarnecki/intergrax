# DG-001C Public Launcher Bootstrap — Diagnostic Visibility Qualification

**Verdict:** PASS / QUALIFIED (limit identified)

**Date:** 2026-09-08

**Branch:** `development`

**Start HEAD:** `263dc3dca00ac6b1e68d7b805d830c58f038bf92`

**Task:** `DG-001C-PUBLIC-LAUNCHER-BOOTSTRAP-QUALIFICATION` — qualification only; no diagnostics-core, LKW production wiring, queue, or launcher rebuild.

**Ancestor check:** `git merge-base --is-ancestor 385ce35b4ba8ac1f1af9f76142a7c52e129a8e36 HEAD` → exit 0.

---

## 1. Verdict

```text
DG-001C PUBLIC LAUNCHER BOOTSTRAP DIAGNOSTIC VISIBILITY = QUALIFIED
Failure visibility = LIMIT IDENTIFIED (pre-HOST-DIAG-3 boundary)
No bypass = VERIFIED
Production regression = NONE
```

**Answer to the architectural question:**

| Question | Answer |
| -------- | ------ |
| Does the official public launcher reach Central Diagnostics on very early bootstrap failure? | **NO** — PRE-PYTHON and PYTHON-BOOTSTRAP failures surface shell stderr / process exit / proof KV only |
| Does the public launcher bypass canonical hosting composition? | **NO** — transport wrappers delegate to shared Python runners; hosted phases use `run_hosted_application` seam |
| Is the gap a wiring defect in the public launcher today? | **NO** — conscious architectural boundary before identity + tenant + publisher prerequisites |
| Is this DG-001B? | **NO** — DG-001B covers worker B3–B5 bootstrap; public launcher is a separate surface |

**DG-001C status after qualification:** `CLOSED / QUALIFIED` (boundary documented; producer slice remains future work per R1 §16 item 3).

**DG-001 overall:** `PARTIALLY ADDRESSED` (B0–B5-only failures before diagnostic prerequisites remain open).

---

## 2. Public entrypoint

**Primary official LKW public launchers (transport-only):**

| Wrapper | Shared runner |
| ------- | ------------- |
| `applications/local_workspace_application/scripts/run-lkw-core-platform-proof-windows.bat` | `run-lkw-core-platform-proof.py` |
| `applications/local_workspace_application/scripts/run-lkw-core-platform-proof-linux.sh` | same |
| `applications/local_workspace_application/scripts/run-lkw-core-platform-proof-macos.sh` | same |
| `applications/local_workspace_application/scripts/run-lkw-product-quickstart-windows.bat` | `run-lkw-product-quickstart.py` |

**Canonical invocation:**

```text
uv run --project applications/local_workspace_application python <shared_runner.py> --os-family <os> --wrapper-id <wrapper> ...
```

Shell wrappers contain **no** Python bootstrap logic, **no** runtime construction, **no** diagnostic publisher wiring.

---

## 3. Canonical startup path

```text
Operator
   │
   ▼
Public wrapper (.bat / .sh)          [PRE-PYTHON boundary]
   │
   ▼
uv run --project … python <runner>   [PYTHON-BOOTSTRAP boundary]
   │
   ▼
Proof orchestrator (core / quickstart)
   │
   ├── docker / HTTP / child subprocess phases
   └── application-hosting phase → run-lkw-hosting-proof.py
           │
           └── pytest live hosting tests → hosted_process_launcher.py
                   │
                   ▼
           run_local_workspace_hosted_application
                   │
                   ▼
           run_hosted_application(profile, event_publisher_factory=?)
                   │
                   ▼
           HostedApplicationSupervisor → HOST-DIAG-3 when product factory wired
```

**Verified:**

- No direct `HostedApplicationEngine` construction in public wrappers.
- No custom diagnostic publisher in proof runners.
- LKW foreground composition root (`hosting/foreground.py`) delegates to platform `run_hosted_application`.
- HOST-DIAG-3 composition uses existing `build_hosted_application_diagnostic_event_publisher` only when product passes **both** `diagnostic_orchestrator` and `diagnostic_tenant_binding`.

**Public launcher default:** `event_publisher_factory=None` → platform observability-only default (same boundary as DG-001A).

---

## 4. Diagnostics boundary

| Stage | Failure examples | Visibility today | Identity / tenant / publisher |
| ----- | ---------------- | ---------------- | ----------------------------- |
| PRE-PYTHON | `uv` missing, bad repo path | Shell echo + exit code | **NONE** |
| PYTHON-BOOTSTRAP | `ModuleNotFoundError`, import errors | Process stderr + exit code | **NONE** stable `application_id` / `instance_id` |
| Post-import proof orchestration | `CoreProofError`, docker/child failure | Proof KV stdout (`failure_reason=…`) | **PARTIAL** after manifest/profile load in some phases |
| Host composition + composed publisher | Engine / supervisor lifecycle failure | Observability → Central Diagnostics when HOST-DIAG-3 wired | **YES** when product supplies binding + orchestrator |

**HOST-DIAG-3 boundary:** first reachable only after `HostedApplicationEventPublisher` construction inside product/host composition — **not** at public launcher PYTHON-BOOTSTRAP.

**Production hosted CLI:** `python -m local_workspace_application.hosting` rejects bare bootstrap (`exit 1`, stderr) — requires activated `ProductionProcessComposition`.

---

## 5. Identity flow

| Field | PRE-PYTHON / PYTHON-BOOTSTRAP | After host publisher wired |
| ----- | ----------------------------- | --------------------------- |
| `application_id` | **NO** | YES — profile / definition |
| `instance_id` | **NO** | YES — supervisor minted |
| `tenant_id` | **NO** | YES — product `HostedDiagnosticTenantBinding` |
| Diagnostic publisher | **NO** | YES — composed factory |

No identity fabrication attempted on public launcher path (PASS).

---

## 6. Failure flow

**In scope (qualified visibility paths):**

| Failure mode | Public launcher path | Central Diagnostics |
| ------------ | -------------------- | ------------------- |
| `uv` missing | Shell / `failure_reason=uv_missing` KV | **NO** |
| Import / module bootstrap error | stderr + non-zero exit | **NO** |
| Proof orchestration error | `CoreProofError` KV | **NO** |
| Hosted phase with default platform publisher | Observability export only | **NO** |
| Hosted phase with product HOST-DIAG-3 factory | Observability → Problem lifecycle | **YES** (canonical seam; not default on public path) |

**Out of scope (not claimed closed):**

- B0–B2 failures before diagnostic prerequisites (separate from DG-001C).
- External docker/CI child spawn before Python host process (platform controller slice).
- Implementing a new public-launcher typed failure producer (R1 §16 item 3 — future slice, not this qualification).

---

## 7. Architectural answers (A–C)

### A. Canonical composition paths

| Check | Result |
| ----- | ------ |
| Uses canonical `run_hosted_application` seam | **YES** |
| Direct runtime creation in public wrappers | **NO** |
| Hosting bypass | **NO** |
| Diagnostics bypass of Central Diagnostics core | **NO** |
| Custom publisher in public launcher | **NO** |
| Custom exception pipeline | **NO** — `CoreProofError` + KV; platform hosting uses existing event publisher |

### B. Bootstrap failure before prerequisites

**YES.** PRE-PYTHON and PYTHON-BOOTSTRAP failures occur before application identity, instance identity, tenant binding, and diagnostic publisher exist. Documented boundary; **no remediation implemented** in this task.

### C. Root cause classification

**Conscious architectural boundary** (missing prerequisites before HOST-DIAG-3), **not** a public-launcher wiring defect. Future typed producer belongs to R1 §16 item 3 — **separate from DG-001B** (worker B3–B5, already qualified R6).

---

## 8. Test evidence

**Canonical test module:** `tests/unit/applications/local_workspace_application/test_public_launcher_bootstrap_diagnostic_qualification.py`

| Scenario | Result |
| -------- | ------ |
| Public wrappers delegate via `uv run --project … python` | PASS |
| Core / quickstart runners omit HOST-DIAG-3 wiring | PASS |
| LKW foreground uses canonical `run_hosted_application` seam | PASS |
| Default public foreground path → no `event_publisher_factory` override | PASS |
| Platform default publisher observability-only | PASS |
| Hosting CLI rejects bootstrap without composition | PASS |
| PYTHON-BOOTSTRAP-class failure → KV visibility, not Central Diagnostics | PASS |

**Related existing suites (unchanged):**

- `tests/unit/applications/local_workspace_application/test_lkw_core_platform_proof.py` (`test_wrapper_thinness`)
- `tests/unit/hosting/test_hosted_default_host_diag_3_qualification.py`
- `applications/local_workspace_application/tests/hosting/test_hosted_foreground.py`

---

## 9. Production changes

**None.** Qualification confirms existing boundaries and canonical seams. Historical launcher success-path fix (`uv --project`) remains; diagnostic visibility before HOST-DIAG-3 is a documented limit, not a regression introduced here.

---

## 10. Confirmations

- Diagnostics core unchanged
- LKW unchanged (no production wiring edits)
- Queue unchanged
- No bypass of Central Diagnostics contracts
- No private API usage in qualification harness
- No worktree / no new branch
