# TOOL-LIFE-X R4 — final root-cause remediation and parent qualification evidence

**Disposition:** `READY FOR INDEPENDENT F8 EXT-L4 AUDIT` (implementation) · **not** parent CLOSED · **not** FRZ promotion · **not** production activation

| Field | Value |
|---|---|
| Operator START_HEAD (F8-Q01/Q02 correction batch) | `fdde0189157fbcdcf71012c24df57ec0350f7352` |
| Operator START_HEAD (EXT-L4 oracle batch) | `49849716c67cc293a692d2e1afe6f5cf1088164c` |
| IMPLEMENTATION_SHA (oracle durable write path) | `85c81f8abf3913b5810845d9def0b859f937141f` |
| IMPLEMENTATION_SHA (F8-Q01/Q02 closure) | Same as **PUBLICATION_HEAD** (single coherent commit: deterministic SQLite fault test + evidence) |
| EXECUTION_SHA | Same as F8-Q01/Q02 **IMPLEMENTATION_SHA** — L1/L2 executed on the candidate tree committed as **PUBLICATION_HEAD** |
| PUBLICATION_HEAD | **Containing Git commit** on `development` for this evidence revision (resolve exact SHA on GitHub: latest commit touching this file, or `git rev-parse HEAD` after sync). Not the pre-Q02 `85c81f8ab…` oracle-only pin. |
| Prior implementation baseline (operator pin) | `10804d0e024e446274b60421a84797ec55e52f88` |
| Lineage (abbrev.) | `10804d0e…` → `85c81f8ab…` (oracle) → `fdde018915…` (evidence SHA mis-pin) → **PUBLICATION_HEAD** (Q01/Q02) |
| Branch | `development` |
| Roadmap authority | `PLATFORM_ENTERPRISE_COMPLETION_ROADMAP.md` §3.0.5 (EC1–EC15 canonical) |
| ADR | [`ADR_ENTERPRISE_TOOL_LIFE_X_CERTIFIED_OWNER_LOSS_AND_STALLED_POLICY.md`](../architecture/ADR_ENTERPRISE_TOOL_LIFE_X_CERTIFIED_OWNER_LOSS_AND_STALLED_POLICY.md) |

## 1. Plan traceability (F0–F8)

| Phase | Disposition |
|---|---|
| F0–F4 | Prior accepted bundles reused; this wave closes systemic RCA defects A–E |
| F5/F6/F7 | Implementation + consolidated L2 on publication SHA |
| F8 EXT-L4 | **READY FOR INDEPENDENT AUDIT** — Cursor does not close parent or FRZ |

## 2. Root-cause summary (BLOCKER E)

**Symptom:** `test_canonical_strict_host_class_b_lost_ack_recovery_single_effect_e2e` hung or surfaced `FrozenInstanceError("cannot assign to field '__traceback__'")` under `production_mode=True`.

**Causes (two defects):**

1. **Frozen governance exceptions:** `@dataclass(frozen=True)` on `ToolGovernanceDeniedError`, `CapabilityNotGrantedError`, and related `RuntimeError` subclasses. Python exception propagation through `contextlib` `finally` (including `canonical_governed_execution_scope`) attempts to attach `__traceback__` to the active exception → `FrozenInstanceError`, masking the real denial (`CapabilityNotGrantedError` for missing `http` capability grant).
2. **Incomplete strict-host composition in qual:** Production path requires explicit `agent_runtime_governance` (capability `http` for `echo`), `canonical_inner_execution_guard`, `DependencyAttemptExecutionBoundary` (`production_mode=True`), and `state.context.config.production_mode=True`. Missing grant triggered (1).

**Remediation:** Remove `frozen=True` from tool-layer governance/MSE exception dataclasses; map `CapabilityNotGrantedError` to traced `ToolGovernanceDeniedError` in `RuntimeToolInvoker._require_agent_runtime_governance`; wire strict production invoker in E2E via `_build_strict_production_invoker_bundle` (reference admission config + materialized boundary + SQLite idempotency ledger).

**Architecture preservation:** No admission bypass, no `production_mode=False` substitute, no alternate executor; canonical owners unchanged.

## 2a. Root-cause summary (BLOCKER D)

**Gap:** Qual HTTP handlers used transient dict counters without durable idempotency semantics, restart proof, or conflict rejection.

**Remediation:** `tool_life_x_r4_class_b_http_provider_oracle.py` — `ClassBHttpProviderOracle` with optional SQLite durability, replay without duplicate effect, payload conflict → 409, empty key → 400, retention envelope fail-closed; handlers via `class_b_http_handler_for_oracle` (lost-ack gated on commit/release events).

## 2b. F8 findings (systemic closure)

| ID | Disposition |
|---|---|
| F8-01 | **Fixed** — qual tool isolated from default HTTP bundle |
| F8-02 | **Fixed** — mandatory physical key + fingerprint |
| F8-A | **Fixed** — `type(handler) is HttpIdempotentPostQualHandler` (not `isinstance`) blocks subclass/tag forgery |
| F8-03 | **Fixed** — UNKNOWN on transport loss after physical admission |
| F8-04 / F8-C | **Qualified** — strict-host E2E with `production_mode=True`, dependency admission, full redrive chain |
| F8-redrive-A/B | **Fixed** — `assert_sanctioned_physical_redrive_dispatch_admitted` + invoker parity |

## 2c. EXT-L4 consolidated correction (BLOCKER D — qualification oracle)

**Symptom (Disposition C):** In-memory cache preceded SQLite commit; per-instance `threading.Lock` did not serialize cross-instance writers; concurrent first-writes could race on `PRIMARY KEY`; durable commit failure could leave a phantom in-memory effect; retention used `len(_memory)` instead of authoritative row count.

**Remediation (qualification-only):**

- SQLite `BEGIN IMMEDIATE` transactions for all durable writes and replays.
- Effect rows committed before in-memory cache update; cache is non-authoritative across instances/processes.
- Retention enforced via `COUNT(*)` on `class_b_effects` before insert.
- `ClassBHttpProviderOraclePersistenceError` on `sqlite3.Error` or corrupt stored JSON → HTTP 503 fail-closed.
- Negative matrix: `test_tool_life_x_r4_class_b_http_provider_oracle_negatives.py` (dual-instance threads, payload conflict race, subprocess spawn replay, corrupt row, retention after restart, HTTP status contract, parallel HTTP single row).

**Production contracts:** unchanged — fix scoped to qualification oracle module and tests.

## 2d. F8-Q01 / F8-Q02 final correction (EXT-L4)

| ID | Issue | Resolution |
|---|---|---|
| F8-Q01 | `test_oracle_commit_failure_no_memory_effect` used `os.chmod(S_IREAD)` — non-portable, non-deterministic SQLite write-failure simulation | Replaced with test-scoped SQLite `BEFORE INSERT` trigger `RAISE(ABORT, 'qual_injected_sqlite_write_failure')` on `class_b_effects` for the rejected key only (`test_oracle_sqlite_insert_failure_rollback_no_phantom_cache`) |
| F8-Q02 | Evidence published @ `fdde018915…` but **PUBLICATION_HEAD** still pointed at oracle implementation `85c81f8ab…` | Reconciled **START_HEAD**, **IMPLEMENTATION_SHA**, **EXECUTION_SHA**, **PUBLICATION_HEAD** semantics in §12; **containing-commit** convention for self-referential publication SHA |

**F8-Q01 proof (write failure / rollback — not COMMIT-time certification):**

1. Seed effect committed (`seed` key).
2. Fault trigger installed for `new-key` INSERT only.
3. `oracle.apply_post` → `ClassBHttpProviderOraclePersistenceError` with `sqlite3.Error` cause.
4. No SQLite row, no `_memory` entry for rejected key; seed row preserved.
5. Trigger dropped (fixture `finally`); retry succeeds once; replay idempotent; fresh oracle instance sees durable effects.

**F8-Q01 residual:** Trigger exercises real SQL INSERT failure inside `BEGIN IMMEDIATE` transaction rollback path; does not inject failure at final `COMMIT` boundary.

**F8 independent acceptance:** **PENDING** — implementer qualification only.

## 3. Defects A–E (implementation)

| ID | Disposition |
|---|---|
| A | Redrive dispatch admission — `sanctioned_physical_redrive_dispatch_admission.py` |
| B | `invoke_sanctioned_physical_redrive` — `_prepare_invocation`, protected-work, `before_external_effect_sanctioned_physical_redrive`, `_execute_once`, `after_external_effect` |
| C | Exact-type qual handler provenance — `class_b_qualification_provenance.py` |
| D | Provider oracle — SQLite-authoritative idempotency + negative matrix |
| E | Production admission E2E green (see §2) |

## 4. Changed owners (this batch)

| Path | Role |
|---|---|
| `intergrax/runtime/external_operations/sanctioned_physical_redrive_dispatch_admission.py` | A — dispatch admission |
| `intergrax/runtime/nexus/tools/invoker.py` | B + E — redrive + governance mapping |
| `intergrax/runtime/tools/idempotency_pre_effect_coordinator.py` | B — sanctioned redrive reconcile |
| `intergrax/contracts/idempotency_store.py` + store impls | B — `reconcile_suspended_pre_effect_for_sanctioned_physical_redrive` |
| `intergrax/tools/providers/http/class_b_qualification_provenance.py` | C — exact handler type |
| `intergrax/runtime/agent_governance/errors.py` | E — exception traceback fix |
| `tests/.../test_tool_life_x_r4_f8_remediation.py` | E2E + oracle smoke |
| `tests/.../test_tool_life_x_r4_class_b_http_provider_oracle_negatives.py` | D — concurrency/durability negatives |
| `tests/.../test_tool_life_x_r4_f8_redrive_dispatch_adversarial.py` | A/B negatives |
| `tests/.../tool_life_x_r4_class_b_http_provider_oracle.py` | D — oracle |

## 5. Physical qualification (L2)

```powershell
uv run pytest `
  tests/unit/runtime/external_operations/test_tool_life_x_r4_f8_remediation.py `
  tests/unit/runtime/external_operations/test_tool_life_x_r4_f8_redrive_dispatch_adversarial.py `
  tests/unit/runtime/external_operations/test_tool_life_x_r4_systemic_rca_authoritative_classification.py `
  tests/unit/runtime/external_operations/test_tool_life_x_r4_b1_authoritative_effect_class.py `
  tests/unit/runtime/external_operations/test_tool_life_x_r4_b2_http_class_b_physical_qual.py `
  tests/unit/runtime/external_operations/test_tool_life_x_r4_b2_mcp_class_c_fail_closed.py `
  tests/unit/runtime/external_operations/test_tool_life_x_r4_ext_l4_binding_adversarial.py `
  tests/unit/runtime/external_operations/test_tool_life_x_r4_ext_l4_integrated_http_redrive_physical.py `
  tests/unit/runtime/external_operations/test_tool_life_x_r4_f6_scenario_24_recovery.py `
  tests/unit/applications/test_tool_life_x_r4_production_entry_e2e_qualification.py `
  tests/unit/runtime/nexus/tools/test_tool_life_x_r4_terminal_lifecycle_invoker.py `
  tests/unit/runtime/external_operations/test_tool_life_x_r4_class_b_http_provider_oracle_negatives.py `
  -q -p no:xdist
```

**Result:** 67 passed @ F8-Q01/Q02 **EXECUTION_SHA** — log `.tmp/session/TOOL-LIFE-X-R4-F8-Q01-Q02/pytest-l2-consolidated.log` (prior EXT-L4: `.tmp/session/TOOL-LIFE-X-R4-F8-EXT-L4/pytest-l2-consolidated.log`)

**L1 (oracle-only):** `uv run pytest -q -p no:xdist tests/unit/runtime/external_operations/test_tool_life_x_r4_class_b_http_provider_oracle_negatives.py` — **8 passed** — `.tmp/session/TOOL-LIFE-X-R4-F8-Q01-Q02/pytest-l1-oracle.log` (`PYTHONHASHSEED=0`, Windows, `uv` pytest)

**L1 command (Q01):** same path as above; environment: local Windows dev, `uv run pytest`, `-p no:xdist`.

**Strict production E2E:** `test_canonical_strict_host_class_b_lost_ack_recovery_single_effect_e2e` — `production_mode=True`, dependency boundary, governance `http` capability, lost-ack, redrive, `oracle.effect_count == 1`.

**EQG:** `uv run ruff check` on changed owners — pass. Full `pytest tests/unit -m "gate and not no_ci"` — **not PASS** (16 pre-existing collection errors, e.g. `tests/unit/tools/providers/rag/test_rag_scope.py`); scoped L2 used as change-impact gate.

**L3 reuse:** SQLite durable revocation helpers in integrated HTTP redrive + strict E2E; no new PostgreSQL/Docker L3 on this delta.

## 6. Closure matrix (excerpt)

| Criterion | Owner | Test / gate | Status |
|---|---|---|---|
| F8-redrive-A | `sanctioned_physical_redrive_dispatch_admission` | `test_tool_life_x_r4_f8_redrive_dispatch_adversarial.py` | VALIDATION REQUIRED @ audit SHA |
| F8-04 production admission | `declarative_tool_wiring` + invoker | strict-host E2E | VALIDATION REQUIRED @ audit SHA |
| BLOCKER D oracle | qual oracle module | oracle negatives + B2 HTTP qual + L2 | **RE-VALIDATED** @ `85c81f8ab…` + Q01 deterministic fault @ **EXECUTION_SHA** |
| F8-Q01 SQLite fault | test trigger INSERT ABORT | L1 oracle negatives | **PASS** @ **EXECUTION_SHA** |
| F8-Q02 evidence SHA | this document §12 | publication lineage | **RECONCILED** @ **PUBLICATION_HEAD** |
| FRZ-* global | platform | EQG full unit gate | BLOCKED (collection errors unrelated) |
| Parent CLOSED | operator F8 | independent audit | OPERATOR DECISION REQUIRED |

## 7. EC1–EC15 (individual rows — evidence pointer)

| EC | Requirement (summary) | Status | Test / evidence |
|---|---|---|---|
| EC1 | Durable external-operation identity | ALREADY PROVEN + extended | strict E2E, invoker bind |
| EC2 | Physical attempt / RUNNING semantics | ALREADY PROVEN | strict E2E, scenario 24 |
| EC3 | Certified owner loss | ALREADY PROVEN | strict E2E, L3 R8 helpers |
| EC4 | Reconciliation + redrive CAS | ALREADY PROVEN | strict E2E, adversarial dispatch |
| EC5 | UNKNOWN / uncertain outcomes | TESTED | F8 remediation, transport loss |
| EC6 | Class B vs C separation | TESTED | B1/B2/MCP tests |
| EC7 | Stalled / abandonment policy | REUSE prior ADR qual | scenario 24 subset |
| EC8 | Worker epoch / revocation | TESTED | strict E2E + sqlite revocation |
| EC9 | Single terminal writer | TESTED | terminal lifecycle invoker |
| EC10 | Resource / DB limits | REUSE F5 | unchanged owners |
| EC11 | MCP Class C default | TESTED | `test_tool_life_x_r4_b2_mcp_class_c_fail_closed` |
| EC12 | HTTP Class B physical qual | PHYSICALLY QUALIFIED (local) | integrated + strict E2E |
| EC13 | Cross-process durability | REUSE L3 R8 | sqlite shared store helpers |
| EC14 | No authority bypass | TESTED | adversarial binding/dispatch |
| EC15 | Lifecycle completeness | TESTED | terminal lifecycle + E2E |

## 8. Scenarios 1–24

Scenarios covered by consolidated L2 (not re-listed exhaustively here): positive lost-ack/redrive (**24**), competing redrive (**24** variants in `test_tool_life_x_r4_f6_scenario_24_recovery.py`), binding adversarial (**EXT-L4**). Full scenario-by-scenario re-proof @ audit SHA — **VALIDATION REQUIRED**.

## 9. G-A01–G-A09

| ID | Status |
|---|---|
| G-A01 | REUSE prior platform evidence |
| G-A02 | TESTED — authoritative Class B qual tool only |
| G-A03–G-A06 | REUSE unless touched |
| G-A07 | TESTED — qual wiring isolated |
| G-A08–G-A09 | REUSE prior |

## 10. Residuals / blockers

| ID | Blocker |
|---|---|
| EQG-global | Full unit gate collection failures (16 errors) — out of R4 diff scope |
| F8 sign-off | Independent EXT-L4 auditor acceptance |
| FRZ promotion | Operator-only |
| Live vendor L3 | Not claimed; local HTTP qual only |

## 11. Conclusion

Implementation and scoped L2 support **READY FOR INDEPENDENT F8 EXT-L4 AUDIT**. **Parent CLOSED not asserted.**

## 12. Publication

| Field | Value |
|---|---|
| START_HEAD (F8-Q01/Q02 batch) | `fdde0189157fbcdcf71012c24df57ec0350f7352` |
| START_HEAD (EXT-L4 oracle batch) | `49849716c67cc293a692d2e1afe6f5cf1088164c` |
| IMPLEMENTATION_SHA (oracle code) | `85c81f8abf3913b5810845d9def0b859f937141f` |
| IMPLEMENTATION_SHA (Q01/Q02) | **PUBLICATION_HEAD** (test + evidence only; oracle production path unchanged) |
| EXECUTION_SHA | **PUBLICATION_HEAD** (L1/L2 run on committed candidate immediately before push) |
| PUBLICATION_HEAD | Git commit on `development` **containing this file revision** (self-referential; auditor resolves exact SHA from GitHub — must not be confused with oracle-only `85c81f8ab…`) |

## 13. F8 EXT-L4 impact matrix (oracle delta)

| Element | Canonical owner | Affected contracts | Prior evidence SHA | Change impact | Historical validity | Required validation | Result | Blocker |
|---|---|---|---|---|---|---|---|---|
| F8-A | `class_b_qualification_provenance` | qual handler type | `10804d0e…` | none | **REUSE** | L2 if oracle touched HTTP bind | **REUSE** — L2 pass | — |
| F8-B | invoker redrive | sanctioned redrive | `10804d0e…` | none | **REUSE** | adversarial L2 | **REUSE** — L2 pass | — |
| F8-C | strict E2E | production_mode admission | `10804d0e…` | indirect (shared oracle in E2E) | partial re-proof | strict E2E L2 | **RE-VALIDATED** | — |
| F8-D | qual oracle | none (test-only) | prior oracle | **SUBSTANTIVE** (Q01 test-only) | chmod negative **INVALIDATED**; trigger negative **PASS** | L1+L2 negatives | **PASS** @ **EXECUTION_SHA** | — |
| F8-Q01 | oracle negatives test | none | chmod test | **SUBSTANTIVE** | prior chmod claim **INVALIDATED** | L1 deterministic INSERT fault | **PASS** | — |
| F8-Q02 | evidence publication | none | `fdde018915…` mis-pin | **DOC** | misleading PUBLICATION_HEAD **INVALIDATED** | §12 reconciliation | **PASS** @ **PUBLICATION_HEAD** | — |
| F8-E | governance exceptions | tool governance errors | `10804d0e…` | none | **REUSE** | L2 | **REUSE** | — |
| EC12 | HTTP Class B physical | trusted effect registry | prior bundles | oracle fixture only | extended | L2 integrated + strict | **PASS scoped** | — |
| EC14 | redrive binding | `bound_redrive_effect_*` | EXT-L4 | none | **REUSE** | adversarial | **REUSE** | — |
| FRZ-* global | platform | EQG full unit | prior | none | **REUSE** | `gate and not no_ci` collect | **BLOCKED** | 16 collection errors (RAG/harness) |
| Parent CLOSED | operator F8 | independent audit | — | — | n/a | external sign-off | **OPEN** | independent EXT-L4 |

Scenarios **1–24**, **G-A01–G-A09**, **EC1–EC15** (except EC12 row above): **REUSE** prior scoped PASS @ `10804d0e…` / EXT-L4 bundles; **no automated full re-run** — consolidated L2 green confirms no regression on touched F8 modules. Independent auditor re-proof @ publication SHA remains **VALIDATION REQUIRED** per scenario row in [`TOOL_LIFE_X_R4_EXT_L4_CORRECTIVE_EVIDENCE_BUNDLE.md`](TOOL_LIFE_X_R4_EXT_L4_CORRECTIVE_EVIDENCE_BUNDLE.md).

**EQG classification:** `F8-SCOPED QUALIFICATION` — 67/67 L2 PASS; `TOOL-LIFE-X PARENT CERTIFICATION` — **PARTIAL** (oracle gap closed, parent gate still OPEN); `GLOBAL PLATFORM EQG/FRZ` — **BLOCKED** (pre-existing collection errors, not introduced by this delta).
