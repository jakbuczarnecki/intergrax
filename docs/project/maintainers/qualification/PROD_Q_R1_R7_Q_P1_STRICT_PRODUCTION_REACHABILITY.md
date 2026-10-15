# PROD-Q-R1-R7-Q-P1 — Strict production reachability gate

| Field | Value |
|-------|-------|
| **Status** | **READY FOR AUDIT** |
| **Parent** | PROD-Q-R1-R7-Q (not PASS) |
| **Branch** | `development` |
| **START_HEAD** | `ac367ea699e3b592272ff74a5a6d3999bf2ea707` |
| **FINAL_COMMIT** | `8ec496603ab2f70b3da5db350c90aba87e2488d4` |
| **Gate implementation commit** | `ae9cfdb564ca2aecff66cdfb017495e7402870a8` |
| **Audited production SHA** | `ccbb19633e83de29d5b70c5e7e1689b0c29d8c07` (boundary code baseline; gate delivered on FINAL_COMMIT) |
| **Method** | Closed-world AST call-site reachability + adversarial snippet negatives; selective D2/D4 reuse |

## Strict production roots (trusted entry set)

| Root symbol | Module |
|-------------|--------|
| `open_strict_production_serving_session` | `intergrax/applications/_shared/strict_production_serving_launcher.py` |
| `create_governed_contractor_strict_production_asgi_app` | `applications/governed_contractor_application/host/strict_production_serving.py` |
| `create_governed_contractor_process_app` | `applications/governed_contractor_application/host/main.py` |
| `create_app` | `applications/governed_contractor_application/host/main.py` |
| `run` | `applications/governed_contractor_application/host/main.py` |

**P1-G01 sanctioned chain (static):** `open_strict_production_serving_session` → `open_production_process_lifecycle` → `wire_production_lifecycle_services` → `reject_reference_runtime_deployment_adapter` → D3 reconciliation / activation handoff (in `production_process_lifecycle` / launcher) → GCA `create_governed_contractor_strict_production_asgi_app` → `create_governed_contractor_process_app`. Legacy `main.run()` remains fail-closed (D4-03 behavioral evidence reused).

## Closed-world boundary modules

1. `applications/governed_contractor_application/host/main.py`
2. `applications/governed_contractor_application/host/strict_production_serving.py`
3. `intergrax/applications/_shared/strict_production_serving_launcher.py`
4. `intergrax/applications/_shared/production_host_composition.py`
5. `intergrax/applications/_shared/strict_production_asgi_lifespan.py`
6. `intergrax/applications/_shared/production_process_lifecycle.py`
7. `intergrax/applications/_shared/production_lifecycle.py`
8. `intergrax/applications/_shared/reference_production_lifecycle.py` (inventory only — not a strict root)
9. `intergrax/applications/_shared/colocated_serving_runtime_composition.py`
10. `intergrax/agent_distribution/deployment.py`

No whole-repository scan. Direct import/callee resolution stays inside this set plus explicit application-package mapping for `governed_contractor_application.host.*`.

## Allowed rejection-only imports (not execution bypass)

| Symbol | Module | Use |
|--------|--------|-----|
| `FakeInMemoryRuntimeDeploymentAdapter` | `production_lifecycle.py` | `isinstance` guard inside `reject_reference_runtime_deployment_adapter` only |
| `ReferenceProductionLifecycleGovernanceBlockedError` | `strict_production_serving_launcher.py` | Exception translation (not reference lifecycle opener) |
| `ReferenceProductionLifecycleServices` / `ReferenceProductionLifecycleResult` | `production_lifecycle.py` | Type aliases / return types on production launcher (no `ReferenceProductionLifecycleLauncher` construction) |

## Prohibited execution paths (gate asserts zero reachable calls)

- `FakeInMemoryRuntimeDeploymentAdapter(...)` construction (not `isinstance` guards)
- `open_reference_production_process_lifecycle(...)`
- `ReferenceProductionLifecycleLauncher(...)`
- `wire_reference_production_lifecycle_services(...)` / `wire_durable_reference_production_lifecycle_services(...)`
- `create_reference_production_process_composition(...)`
- Deployment adapter construction inside GCA strict ASGI factory modules (`strict_production_serving.py`, GCA `main.py` process factory path)

## Static gate coverage

| Artifact | Role |
|----------|------|
| `tests/qualification/prod_q/_prod_q_r1_r7_q_p1_strict_reachability.py` | Bounded AST indexer, BFS over resolved calls from strict roots |
| `tests/qualification/prod_q/test_prod_q_r1_r7_q_p1_strict_production_reachability.py` | P1-G01–G06 assertions + adversarial snippet negatives |

**Coverage boundary (explicit):**

- Proves absence of **listed prohibited direct calls** on the **bounded static call graph** reachable from strict roots through same-module calls and import-resolved cross-module calls.
- Does **not** prove full interprocedural call-graph closure: attribute dispatch (`lifecycle.launcher.deploy_and_activate`), higher-order callbacks, dynamic imports, or reflection are **out of scope**.
- Does **not** claim global `FakeInMemory*` count = 0 outside the closed-world module set.

**Unresolved-edge policy:** Critical imported authority calls (`open_production_process_lifecycle`, `wire_production_lifecycle_services`, `create_governed_contractor_process_app`, etc.) must resolve to a boundary module or the gate fails closed.

## P1-R1 gate hardening (PROD-Q-R1-R7-Q-P1-R1)

| Defect (before) | Correction |
|-----------------|------------|
| P1-G05 used text presence (`ready_for_traffic` in file) | AST guard: `if not session.readiness.ready_for_traffic` must `raise` before any `create_governed_contractor_process_app` call; snippet negatives for docstring-only and neutralized guards |
| GCA deployment ctor matched local name `== suffix` only | Resolve import bindings + qualified attributes (`AlternateAdapter()`, `deployment_mod.FakeInMemory…()`) via shared `_deployment_adapter_constructor_target`; preserve `isinstance` rejection allowance |

## P1-R2 ASGI guard branch correctness (PROD-Q-R1-R7-Q-P1-R2)

| Defect (before) | Correction |
|-----------------|------------|
| P1-G05 `ast.walk(if)` treated `else` and nested `if` bodies as not-ready proof | Analyze only `if` body (not `orelse`); require top-level `raise` (no nested `if`); reject `process_app` on body or before guard floor lineno |
| Raise in `else` or optional nested `if` implied admission | Snippet negatives: raise-only-in-else, nested optional `if`, process app before guard, empty `pass` body |

Behavioral D4 reuse: `test_d4_asgi_startup_only_after_preconditions` (unready session → `ValueError`) — run under default `conftest` when session start succeeds; P1 module itself requires `--noconftest` at this HEAD (see below).

## Adversarial negatives (snippet fixtures)

| Case | Expected |
|------|----------|
| `FakeInMemoryRuntimeDeploymentAdapter()` in synthetic module | **Detect** |
| `FakeInMemoryRuntimeDeploymentAdapter as AlternateAdapter` then `AlternateAdapter()` | **Detect** |
| `deployment_mod.FakeInMemoryRuntimeDeploymentAdapter()` qualified call | **Detect** |
| `isinstance(..., FakeInMemoryRuntimeDeploymentAdapter)` in reject helper | **Allow** |
| `open_reference_production_process_lifecycle()` call | **Detect** |
| ASGI factory: docstring mentions `ready_for_traffic` only | **Fail** structural guard |
| ASGI factory: `create_governed_contractor_process_app` on not-ready branch | **Fail** |
| ASGI factory: `if not ready_for_traffic: pass` without `raise` | **Fail** |
| ASGI factory: `raise` only in `else` of ready guard | **Fail** |
| ASGI factory: `raise` only inside nested optional `if` on not-ready body | **Fail** |
| ASGI factory: `create_governed_contractor_process_app` before guard | **Fail** |

## Pytest evidence

**Command (sequential, no xdist):**

```text
uv run pytest -p no:xdist -q --noconftest \
  tests/qualification/prod_q/test_prod_q_r1_r7_q_p1_strict_production_reachability.py \
  tests/unit/applications/test_prod_q_r1_r7_d2_production_process_lifecycle.py::test_d2_production_process_lifecycle_module_has_no_fake_adapter_reachability \
  tests/unit/applications/governed_contractor/test_prod_q_r1_r7_d4_strict_production_gca_launcher.py::test_d4_fake_adapter_rejected \
  tests/unit/applications/governed_contractor/test_prod_q_r1_r7_d4_strict_production_gca_launcher.py::test_d4_03_legacy_main_run_fail_closed \
  tests/unit/applications/governed_contractor/test_prod_q_r1_r7_d4_strict_production_gca_launcher.py::test_d4_asgi_startup_only_after_preconditions
```

**Result @ workspace (P1-R1):** 20 passed @ `--noconftest` (~0.2 s); D4 `test_d4_asgi_startup_only_after_preconditions` 1 passed with default conftest (~3 s). Log: `.tmp/session/prod-q-r1-r7-q-p1-r1/p1-only.log`.

**Result @ workspace (P1-R2):** 23 passed @ `--noconftest` (~0.2 s) on `test_prod_q_r1_r7_q_p1_strict_production_reachability.py` only. Log: `.tmp/session/prod-q-r1-r7-q-p1-r2/p1-only.log`.

**ENVIRONMENT/TEST ISSUE — EVIDENCE REQUIRED:** Default `tests/conftest.py` `pytest_sessionstart` triggers a circular import (`RuntimeCheckpoint` ↔ `agent_runtime_io` via `intergrax.runtime.long_running` / `agent_distribution`) at HEAD `ec773c666…` when collecting `tests/qualification/prod_q/…` (INTERNALERROR before collection). **Not** ordinary freeze debt. **Separate corrective dependency (pre–final R7-Q certification):** break the `RuntimeCheckpoint` import cycle (or defer `pytest_sessionstart` model_rebuild) so qualification and default CI pytest entry collect without `--noconftest`. Do not treat `--noconftest` as proof that the default qualification entry works.

## P1-DEFAULT-PYTEST qualification run (`PROD-Q-R1-R7-Q-P1-DEFAULT-PYTEST-EVIDENCE-RECORD`)

| Field | Value |
|-------|-------|
| **run_id** | `20261015-161228-utc` |
| **EXECUTION_HEAD** | `036912bec45ac627b6c99411c82db72f75df858f` |
| **Authorized invocations** | Exactly **one** pytest process (zero retry) |
| **Command** | `uv run pytest -p no:xdist -q tests/qualification/prod_q/test_prod_q_r1_r7_q_p1_strict_production_reachability.py` |
| **Default `tests/conftest.py`** | **Enabled** (no `--noconftest`) |
| **Capture self-check** | **PASS** (`STDOUT_CAPTURE_OK` / `STDERR_CAPTURE_OK`) |
| **UTC start** | `2026-10-15T14:12:28.516Z` |
| **UTC end** | `2026-10-15T14:12:32.607Z` |
| **Wall (pytest)** | 4091 ms |
| **Exit code** | 0 |
| **Timeout** | None (wait ceiling 300000 ms; `timeout_result` = false) |
| **Summary** | `23 passed in 0.51s` |
| **Failed / skipped / xfail** | 0 / 0 / 0 |
| **Raw evidence** | `.tmp/proof/R7-Q-P1-DEFAULT-PYTEST-20261015-161228-utc/` (`pytest.stdout.log`, `pytest.stderr.log`, `capture-selfcheck.*.log`, `run.meta.json`) |
| **P1-G01–G06** | **PASS** (bounded static gate on this module only) |
| **Coverage boundary** | Same as §Static gate coverage — **not** a full dynamic/HTTP production bypass proof; D2/D4 reuse **not** re-run in this invocation |

**Root-cause disposition:** Historical default-`conftest` collection failure documented above @ `ec773c666…` was **not reproduced** at **EXECUTION_HEAD**; **no code repair warranted** for this evidence line.

**FRZ local evidence contributions (global statuses unchanged):** FRZ-PRD-01/03/07/08, FRZ-SEC-04, FRZ-REG-02/03/09 — same bounded static P1 intent as §FRZ contributions; this run adds default-conftest qualification-entry replay only.

**Parent / wave posture (unchanged):** **P1** = **READY FOR AUDIT**; **P2** / **P3** = **CLOSED — scoped**; **E.1** = **CLOSED — scoped**; **PROD-Q-R1-R7-Q** = **IN PROGRESS — NOT QUALIFIED**; **PROD-Q-R1** / **PROD-Q** = **NOT CLOSED**.

## Reused D2/D4 evidence (not duplicated)

- D2 `test_d2_production_process_lifecycle_module_has_no_fake_adapter_reachability` — `production_process_lifecycle.py` text negative
- D4 `test_d4_fake_adapter_rejected` — runtime rejection on strict launcher
- D4 `test_d4_03_legacy_main_run_fail_closed` — `main.run()` fail-closed
- D4 `test_d4_asgi_startup_only_after_preconditions` — unready session rejected

## Enterprise audit matrix (P1 scope only)

| Criterion | Verdict |
|-----------|---------|
| Contracts over implementations | PASS — gate checks sanctioned ports/call surfaces, not adapter internals |
| Hard layer boundaries | PASS — no production code changes; inventory respects tier boundaries |
| Exactly-one semantic owner | PASS — strict opener / lifecycle ownership unchanged |
| Exactly-one composition owner | PASS — GCA consumes `ProductionProcessComposition`; no new composition paths |
| Strong typing | N/A — test-only delta |
| A1 structural constraints | PASS — no bypass of VENV strict pin in changed scope |
| Fail-closed | PASS — unresolved critical calls and prohibited calls fail tests |
| Governance ≠ execution | PASS — no new activation authority |
| Traceability | PASS — regression module + this evidence doc |

## Tenant isolation audit (P1 changed scope only)

**Verdict:** No change to tenant binding or resolver behavior in P1 deliverables (tests/docs only). **Not** a global tenant certification.

## FRZ contributions (evidence only — global statuses unchanged)

| FRZ | Contribution |
|-----|----------------|
| FRZ-PRD-01 | Strict single-root serving chain documented + static regression |
| FRZ-PRD-03 | Production lifecycle wiring reachability guarded |
| FRZ-PRD-07 | Reference/fake execution paths excluded on bounded strict graph |
| FRZ-PRD-08 | No harness/reference lifecycle calls from strict roots (static) |
| FRZ-SEC-04 | Fail-closed legacy entry reinforced via D4 reuse |
| FRZ-REG-02/03/09 | Registry projection bootstrap path remains store-backed; no seeding calls on strict graph |

## Findings

| ID | Disposition |
|----|-------------|
| IN-SCOPE BLOCKER | None |
| TRACKED FREEZE DEBT | R7-Q integrated E2E, cross-tenant pack, full conftest pytest path — unchanged |
| ENVIRONMENT/TEST ISSUE | Prior report @ `ec773c666…` preserved; @ `036912bec45ac627b6c99411c82db72f75df858f` default-conftest P1 module run `20261015-161228-utc` **PASS** — historical failure **not reproduced**; no corrective code required for this line |

## Recommended R7-Q disposition

- **PROD-Q-R1-R7-Q:** remain **BLOCKED** / not PASS (P1 is one gate only).
- **Next bounded child:** **PROD-Q-R1-R7-Q-P2** — integrated strict-path qualification (per gate plan §E), including HTTP/runtime evidence where authorized.

---

Wprowadzone zmiany muszą zostać niezależnie zaudytowane na podstawie kodu z commitu znajdującego się na GitHubie. Raport Cursor AI nie jest podstawą do finalnego zamknięcia zadania.
