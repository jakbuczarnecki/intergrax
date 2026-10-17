# PROD-Q-R1-R7-D3 — Durable projection inputs & post-restart reconciliation

| Field | Value |
|---|---|
| **Status** | **READY FOR AUDIT** (implementation evidence — does not close PROD-Q-R1-R7, PROD-Q, or FRZ-PRD global PASS) |
| **Parent** | PROD-Q-R1-R7 |
| **Successor** | PROD-Q-R1-R7-D4 |
| **START_HEAD** | `b631c18cf270bde29264d5e83d0ebb631173f0dc` |
| **DEC anchors** | DEC-04 (durable AD + descriptor authority), DEC-08 (physical attestation before serving reliance) |

## Design summary

- **Durable projection inputs:** `DescriptorBackedRegistryProjectionInputStore` implements `RegistryProjectionInputStore` by reconstructing `RegistryProjectionInputBundle` from `RuntimeRegistryProjectionDescriptor` (SQLite) plus `RegistryProjectionAuthorityResolver` (revision, roster, lock, materialization). Session `register()` retains live resolver instances; `get()` rebuilds factory resolver via `assemble_production_registry_projection_input_bundle` — no callable pickle.
- **Production wiring:** `wire_production_lifecycle_services` and `wire_durable_reference_production_lifecycle_services` use the descriptor-backed store.
- **Restart reconciliation:** `ActivationService.reconcile_traffic_serving_physical_unit_after_process_restart` attests traffic-serving deployment instances (READY/SERVING). `reconcile_production_serving_environment_on_process_open` composes attestation, projection input reconstruction, rehydration, and coordinator `prepare_projection`. `open_production_process_lifecycle` unions durable `ApplicationEnvironmentServingStore.list_traffic_serving_records()` with caller hints via `resolve_mandatory_production_serving_reconciliation_targets` (caller cannot omit active traffic). Fails closed on reconciliation errors before committing the process slot.
- **D3-R1 (B02):** `DescriptorBackedRegistryProjectionInputStore.register` validates durable descriptor authority before session publication; `get` prefers durable reconstruction when a descriptor exists (staged-only path when descriptor absent).

## Evidence commands

```text
uv run pytest -p no:xdist tests/unit/applications/test_prod_q_r1_r7_d3_r1_recovery_projection_integrity.py tests/unit/applications/test_prod_q_r1_r7_d3_production_restart_reconciliation.py tests/unit/applications/test_prod_q_r1_r7_d2_production_process_lifecycle.py tests/unit/applications/test_registry_projection_descriptor_corruption.py tests/integration/agent_distribution/test_enterprise_agent_lifecycle_durable_e2e.py tests/integration/agent_distribution/test_enterprise_projection_rehydration_e2e.py
uv run pyright intergrax/applications/_shared/durable_registry_projection_input_store.py intergrax/applications/_shared/production_process_lifecycle.py intergrax/applications/_shared/production_process_serving_reconciliation.py intergrax/agent_distribution/stores.py intergrax/agent_distribution/in_memory_stores.py intergrax/agent_distribution/sqlite_stores.py
```

## D2 alignment note

Operator disposition: D2 remains **ACCEPTED / READY FOR AUDIT** within A1; this stage does not reopen D2 implementation.

---

## Current-HEAD independent acceptance audit (Cursor)

| Field | Value |
|-------|-------|
| **Task** | `PROD-Q-R1-R7-D2-D3-CURRENT-HEAD-INDEPENDENT-ACCEPTANCE` (D3 slice) |
| **AUDITED_HEAD** | `a7e959d2d3171924c506ab9483e466543aef39b1` |
| **Implementation tip** | `f2dd8ecbbff8561a50ec10321ddef62d345b65a9` (D3-R1) — **identical** to AUDITED_HEAD for D3-owned modules |
| **Production delta** | **0** (audit-only) |

### Drift vs D3-R1 evidence SHA

No file changes on `durable_registry_projection_input_store.py`, `production_process_serving_reconciliation.py`, or D3 reconciliation hooks in `production_process_lifecycle.py` since `f2dd8ecbb…`. Post-D3-R1 work (**DIM13**, **DIM16**, D4 launcher) composes **through** `open_production_process_lifecycle` / reconciliation without altering D3 semantic owners.

### D3 criterion matrix (@ HEAD)

| Label | Map | Result | Evidence |
|-------|-----|--------|----------|
| D3-01 | durable descriptor authority | **PASS** | D3-R1 register validation + mismatch negatives |
| D3-02 | reconstruction continuity | **PASS** | descriptor-backed reconstruct + SQLite restart test |
| D3-03 | physical liveness reconciliation | **PASS** | `reconcile_traffic_serving_physical_unit_after_process_restart` tests |
| D3-04 | no stale serving resurrection | **PASS** | stale physical / empty adapter / mandatory targets |
| D3-05 | recovery sequencing | **PASS** | open-path reconciliation before slot commit; coordinator `prepare_projection` in pipeline test |
| D3-06 | regression / error continuity | **PARTIAL** | D3-R1 + D3 negatives **PASS**; `test_registry_projection_descriptor_corruption.py` **FAIL** at harness setup (see below) |

### Targeted regression (@ AUDITED_HEAD)

```text
uv run pytest -p no:xdist -q \
  tests/unit/applications/test_prod_q_r1_r7_d3_r1_recovery_projection_integrity.py \
  tests/unit/applications/test_prod_q_r1_r7_d3_production_restart_reconciliation.py \
  tests/unit/applications/test_registry_projection_descriptor_corruption.py
```

**19 passed, 1 failed** (exit **1**), ~15s cumulative with D2 module in same run — log: `.tmp/session/PROD-Q-R1-R7-D2-D3-CURRENT-HEAD-INDEPENDENT-ACCEPTANCE/pytest.log`

**Failure:** `test_descriptor_corruption_fail_closed` — `EnterpriseAgentLifecycleProofStack.run_happy_path()` → `AgentBinding requires agent_type or import_path` in `agent_resolution.py` (test-support / catalog binding drift; **not** D3 production module delta).

**DIM13/DIM14:** not replayed in this task (qualification gates under `tests/qualification/prod_q/`); D3 unit scope covers mandatory reconciliation and descriptor-store paths. Integrated DIM13 evidence remains at historical §C SHAs per roadmap.

### Unresolved finding

| Class | Item | Action |
|-------|------|--------|
| **ENVIRONMENT/TEST ISSUE — EVIDENCE REQUIRED** | Rehydrator corruption regression via `EnterpriseAgentLifecycleProofStack` | Bounded harness/catalog binding repair + re-run **one** test module; **not** a D3 production invariant violation |

### Risk review (D3-relevant)

Risks 6–14 from task brief: **PASS** via D3-R1 mandatory targets, stale physical negatives, reconciliation-before-slot, tenant/store binding in D2 module where shared. Risk 15 (DIM16 shutdown): **no D3 authority expansion** — physical admission close is downstream of D3 reconciliation contract.

### FRZ contribution (scoped; global rows **OPEN**)

FRZ-PRD-12, supporting FRZ-REC/FRZ-STA/FRZ-TEN via recovery fail-closed semantics — evidence contribution only.

### D3 acceptance recommendation

**READY FOR AUDIT** — core D3/D3-R1 invariants **proven @ HEAD** (19/19 D3 unit tests). Recommend independent closure review on GitHub SHA `a7e959d2…`; **not** CLOSED until operator accepts and optional corruption-module replay is green. **No D3 production implementation** required for current-HEAD alignment.

---

## D3-R1 descriptor corruption harness binding corrective (Cursor)

| Field | Value |
|-------|-------|
| **Task** | `PROD-Q-R1-R7-D3-R1-DESCRIPTOR-CORRUPTION-HARNESS-BINDING-CORRECTIVE` |
| **START_HEAD** | `ffda0c36180a4603810d92c228252f7fd813d5de` |
| **Production delta** | **0** (test-support / qualification evidence only) |

### Root cause

`reference_install_contract_authority_records(manifest)` → `materialize_manifest_contract_authority_lab_compat` → `resolve_agent_contract_from_binding` in `install_from_catalog` (`canonical_agent_lifecycle_composition.py`). Manifest roster used `AgentBinding(contract_id=…, builder_key=…)` only — no `agent_type` / `import_path` — so `resolve_agent_type` in `intergrax/applications/_shared/agent_resolution.py` raised before install completed.

### Corrected harness contract

| Layer | Before | After |
|-------|--------|--------|
| Manifest roster (projection) | `contract_id` + `builder_key` only | unchanged via `_stage15_lab_manifest_agent_bindings` (revision-bound factory via roster `builder_key`) |
| Install `package_contract_authority` | derived from incomplete manifest binding | `_stage15_install_package_contract_authority_records` using `CanonicalPingAgent().get_contract()` + `package_contract_authority_record` |
| Package metadata | empty `declared_contracts` | `_stage15_project_metadata` with `AgentPackageContractDeclaration` for canonical ping |
| Enterprise `run_happy_path` | full canonical execute | `verify_execution=False` (descriptor corruption needs activation + descriptor only) |
| Canonical `execute_canonical` | host without revision-bound contract authority | `roster_contract_authority_for` + strict host prerequisites (collaborative work sqlite, continuation store, lab decision policy, test GEP) |

### Changed files

- `testing_support/canonical_agent_lifecycle_composition.py`
- `testing_support/enterprise_agent_lifecycle_composition.py`
- `docs/project/maintainers/qualification/PROD_Q_R1_R7_D3_DURABLE_PROJECTION_RESTART_RECONCILIATION.md`

### Regression (@ task close — local)

```text
uv run pytest -p no:xdist -q tests/unit/applications/test_registry_projection_descriptor_corruption.py::test_descriptor_corruption_fail_closed
uv run pytest -p no:xdist -q tests/unit/applications/test_prod_q_r1_r7_d3_r1_recovery_projection_integrity.py tests/unit/applications/test_prod_q_r1_r7_d3_production_restart_reconciliation.py tests/unit/applications/test_registry_projection_descriptor_corruption.py
```

**20 passed** (corruption module included), logs under `.tmp/session/PROD-Q-R1-R7-D3-R1-DESCRIPTOR-CORRUPTION-HARNESS-BINDING-CORRECTIVE/`.

### D3-06 / matrix update

| D3-06 | **PASS** (corruption fail-closed module green @ evidence SHA) |

### Remaining blockers

| Class | Item |
|-------|------|
| *(superseded)* | Canonical lifecycle execution regression — see corrective section below |

### FRZ contribution (scoped)

Supports **FRZ-PRD-12** descriptor fail-closed replay evidence; **FRZ-REC-*** / **FRZ-REG-*** via corruption negatives — no global FRZ PASS.

---

## D3-R1 canonical lifecycle execution regression corrective (Cursor)

| Field | Value |
|-------|-------|
| **Task** | `PROD-Q-R1-R7-D3-R1-CANONICAL-LIFECYCLE-EXECUTION-REGRESSION` |
| **START_HEAD** | `49bde1a1e2d0a326598c76647db6ff17828bbe17` |
| **Classification** | **A** (harness strict-host composition) + **minimal runtime emission alignment** (`context_assembly.v4` in UAEP; existing `runtime_event_payload_policy` SSOT — not a new schema) |

### Root cause (evidence)

`tests/integration/agent_distribution/test_canonical_agent_lifecycle_e2e.py::test_canonical_agent_lifecycle_reaches_serving_and_execution`

1. **Harness (A):** `_stage15_proof_environment` used `product_defaults` with `prompt_defense_enabled=True`. `PromptDefenseMiddleware` blocks `BEFORE_CONTEXT_BUILD` when `HookContext.payload` is not `LlmInferenceHookPayload` (UAEP reflex path in `uaep_executor.py`). Violation publish then hit `RuntimeEventSchemaError`: `phase mismatch for domain_signal: expected step_execution, got context_building` (`hook_runtime_guard._emit_hook_violation` passes `ctx.phase` over catalog `STEP_EXECUTION`).
2. **Harness (A):** Strict host assembly without injected sqlite collaborative-work repos attempted PostgreSQL materialization; without explicit `DecisionRequirementPolicy` / `ExecutionContinuationStateStore` strict composition failed closed.
3. **Emission drift:** `UaepExecutor._emit_context_assembled` emitted `context_assembly.v2` while durable validation requires `context_assembly.v4` (`runtime_event_payload_policy.py`).

### Minimal corrective

| Layer | Change |
|-------|--------|
| `_stage15_proof_environment` | Disable `prompt_defense_enabled` only (retain other strict product defaults + OTEL + dependency concurrency admission) |
| `execute_canonical` | Keep revision-bound `agent_contract_authority`; sqlite `collaborative_work_repositories`; `lab_orchestration_decision_requirement_policy()`; durable continuation store — drop redundant explicit MSE wiring (harness resolves from repos + policy) |
| `uaep_executor._emit_context_assembled` | `ContextAssemblyPayloadV4` (align with `record_context_assembled_from_engine`) |

### Dependency graph (execute host)

```text
install contract authority (prior corrective)
  → serving projection + roster_contract_authority_for
  → build_harness_host_runtime(
        REVISION_BOUND registry,
        sqlite collaborative work,
        lab DecisionRequirementPolicy,
        continuation store,
        stage15 env w/o prompt defense @ BEFORE_CONTEXT_BUILD)
  → UAEP execute → CONTEXT_ASSEMBLED v4
```

### Changed files

- `testing_support/canonical_agent_lifecycle_composition.py`
- `intergrax/runtime/nexus/uaep/uaep_executor.py`
- `docs/project/maintainers/qualification/PROD_Q_R1_R7_D3_DURABLE_PROJECTION_RESTART_RECONCILIATION.md`

### Regression (@ task close — local)

```text
uv run pytest -p no:xdist -q tests/integration/agent_distribution/test_canonical_agent_lifecycle_e2e.py::test_canonical_agent_lifecycle_reaches_serving_and_execution --tb=short
uv run pytest -p no:xdist -q tests/unit/applications/test_registry_projection_descriptor_corruption.py::test_descriptor_corruption_fail_closed --tb=short
uv run pytest -p no:xdist -q tests/unit/applications/test_prod_q_r1_r7_d3_r1_recovery_projection_integrity.py tests/unit/applications/test_prod_q_r1_r7_d3_production_restart_reconciliation.py tests/unit/applications/test_registry_projection_descriptor_corruption.py
```

**2 passed** (canonical lifecycle parametrized), **21 passed** (D3 + corruption), logs under `.tmp/session/PROD-Q-R1-R7-D3-R1-CANONICAL-LIFECYCLE-EXECUTION-REGRESSION/`.

### D3-01…D3-06 disposition

Unchanged for production D3 scope; **D3-06** corruption module remains **PASS**; canonical lifecycle execution proof **PASS** @ evidence SHA (audit gate).

### Remaining / tracked

| Class | Item |
|-------|------|
| **TRACKED FREEZE DEBT** | `hook_runtime_guard` platform hook violations use `ctx.phase` vs catalog `STEP_EXECUTION` — latent when prompt defense enabled on UAEP reflex path |
| **TRACKED FREEZE DEBT** | UAEP `BEFORE_CONTEXT_BUILD` hook shape vs `PromptDefenseMiddleware` / guardrail input scan expectations under strict product security |

### FRZ contribution (scoped)

**FRZ-PRD-12** lifecycle + descriptor evidence; **FRZ-REG-*** regression gates green — no global FRZ PASS.

---

## D3-R1 UAEP event and hook boundary qualification (Cursor)

| Field | Value |
|-------|-------|
| **Task** | `PROD-Q-R1-R7-D3-R1-UAEP-EVENT-AND-HOOK-BOUNDARY-QUALIFICATION` |
| **Parent** | `PROD-Q-R1-R7-D3` |
| **START_HEAD / AUDITED_HEAD** | `7400c345127a54b03b411de6044d290ad70d911a` |
| **Production delta** | **0** (qualification + contract tests only) |
| **Max status** | **READY FOR AUDIT** |

### Gate A — `ContextAssemblyPayloadV4` / `CONTEXT_ASSEMBLED`

| Check | Result |
|-------|--------|
| Policy SSOT requires `context_assembly.v4` | **PASS** — `runtime_event_payload_policy.py` |
| UAEP producer emits V4 + `CONTEXT_BUILDING` phase | **PASS** — `uaep_executor._emit_context_assembled` |
| Authoritative validation on durable write boundary | **PASS** — `assert_runtime_event_schema` → `assert_canonical_production_runtime_event_payload` (`validating_runtime_event_store`) |
| Trace/correlation on UAEP emission | **PASS** — `correlation_id=ctx.correlation_id or ctx.task_id` unchanged |
| Historical v2 consumers (codec) | **NOT REGRESSED** — `spine_payload_codec` still decodes v1–v4; write path strict v4 only |
| Gate A overall | **PASS** |

**Ownership:** producer `UaepExecutor._emit_context_assembled`; payload policy owner `runtime_event_payload_policy`; validation owner `schema_guard` + `payload_registry`.

### Gate B — Prompt Defense / hook violation boundary

| Check | Result |
|-------|--------|
| Why non-`LlmInferenceHookPayload` at `BEFORE_CONTEXT_BUILD` | UAEP reflex builds `HookContext` with `runtime_state` only (`uaep_executor.py`); graph engine path uses `LlmInferenceHookPayload` (`context_manager.py`) |
| `PromptDefenseMiddleware` scope | **PASS (contract)** — only `BEFORE_CONTEXT_BUILD`; requires `LlmInferenceHookPayload` (`application_security_wiring.py`) |
| Mismatch handling | **Fail-closed BLOCK** (not alternate typed boundary on UAEP path) |
| `hook_runtime_guard` phase vs catalog | **FAIL** — `_emit_hook_violation` passes `phase=ctx.phase` (`CONTEXT_BUILDING`); `event_catalog` maps `DOMAIN_SIGNAL` → `STEP_EXECUTION`; durable validation raises `RuntimeEventSchemaError` |
| Publish failure vs security evidence | **PARTIAL** — execution still blocked; violation audit event may not persist when validating store/bus path enforces schema |
| Strict product security + UAEP | **NOT PROVEN** — harness disables `prompt_defense_enabled` only for stage15 proof (`canonical_agent_lifecycle_composition.py`); not production equivalence |
| Gate B overall | **FAIL** (architecture decision required before production closure) |

**Decision gate:** **STOP — ARCHITECTURE DECISION REQUIRED** (child ADR recommended: align UAEP hook payload with `LlmInferenceHookPayload` *vs* extend Prompt Defense to `ContextBuildHookPayload` *vs* catalog phase resolution for platform hook signals *vs* exempt reflex agents — recommendation: **UAEP payload alignment** with graph path per existing guardrail/Prompt Defense contract and `ContextBuildHookPayload` as Tier-0 alternate only if middleware contract is explicitly widened).

### Source / caller inventory (scoped)

| Artifact | Role |
|----------|------|
| `intergrax/runtime/nexus/uaep/uaep_executor.py` | UAEP hook + `CONTEXT_ASSEMBLED` producer |
| `intergrax/runtime/nexus/context/context_manager.py` | Reference `BEFORE_CONTEXT_BUILD` typed payload |
| `intergrax/runtime/events/runtime_event_payload_policy.py` | Event payload SSOT |
| `intergrax/runtime/events/payloads/canonical.py` | `ContextAssemblyPayloadV4` |
| `intergrax/runtime/middleware/hook_runtime_guard.py` | Hook violation `DOMAIN_SIGNAL` publisher |
| `intergrax/applications/_shared/application_security_wiring.py` | `PromptDefenseMiddleware` |
| `intergrax/contracts/middleware_hook_semantics.py` | `LlmInferenceHookPayload`, `ContextBuildHookPayload` |
| `testing_support/canonical_agent_lifecycle_composition.py` | Stage15 harness (prompt defense off for UAEP proof) |

### Test evidence (@ AUDITED_HEAD)

```text
uv run pytest -p no:xdist -q \
  tests/unit/runtime/nexus/test_uaep_context_assembled_event_contract.py \
  tests/unit/applications/test_application_security_wiring.py::test_prompt_defense_blocks_before_context_build_without_llm_payload \
  tests/unit/runtime/middleware/test_hook_runtime_guard.py::test_hook_violation_domain_signal_with_context_building_phase_fails_schema_guard \
  tests/unit/runtime/events/test_runtime_event_payload_policy.py \
  tests/integration/agent_distribution/test_canonical_agent_lifecycle_e2e.py::test_canonical_agent_lifecycle_reaches_serving_and_execution
```

Log: `.tmp/session/PROD-Q-R1-R7-D3-R1-UAEP-EVENT-AND-HOOK-BOUNDARY-QUALIFICATION/pytest-final.log` (**10 passed**)

### Parent blocking

| Parent | Disposition |
|--------|-------------|
| **PROD-Q-R1-R7-D3** (durable projection / restart) | **READY FOR AUDIT** — production D3 scope unchanged |
| **Strict lifecycle + product prompt defense on UAEP** | **BLOCKED** until Gate B ADR + implementation |

### FRZ (scoped; global **OPEN**)

**FRZ-PRD-12**, **FRZ-SEC-01/04/07**, **FRZ-REG-02/03** — evidence contribution only; **new global FRZ PASS = 0**.

### Unresolved findings

| Class | Item |
|-------|------|
| **ARCHITECTURE DECISION REQUIRED** | UAEP `BEFORE_CONTEXT_BUILD` hook shape vs `PromptDefenseMiddleware` under `prompt_defense_enabled=True` |
| **TRACKED FREEZE DEBT** | `hook_runtime_guard` `DOMAIN_SIGNAL` phase override vs `event_catalog` |
| **QUALIFICATION GAP** | Canonical lifecycle E2E does not exercise active Prompt Defense on UAEP path |

---

## IMPL-01 UAEP strict hook security (Cursor; child ADR A+E)

| Field | Value |
|-------|-------|
| **Task** | `PROD-Q-R1-R7-UAEP-STRICT-HOOK-SECURITY-IMPL-01` |
| **ADR** | [`PROD_Q_R1_R7_UAEP_STRICT_HOOK_SECURITY_ADR_01.md`](PROD_Q_R1_R7_UAEP_STRICT_HOOK_SECURITY_ADR_01.md) — **ACCEPTED — OPERATOR RATIFIED A+E** |
| **START_HEAD** | `84c1e550fd847ba9e19bcc87742ca086e888bf2a` |
| **FINAL_COMMIT** | *(set at push)* |

### Historical failure evidence (preserved)

D3-R1 UAEP hook boundary qualification documented **Gate B FAIL** (untyped UAEP payload, harness prompt-defense disable, `hook_runtime_guard` phase mismatch). See § **D3-R1 UAEP event and hook boundary qualification** above.

### Implementation delta

| Gate | Change |
|------|--------|
| **B1** | `uaep_executor.py` — `LlmInferenceHookPayload(prompt=request.message or None)` + tenant subject facet |
| **B2** | `hook_runtime_guard.py` — catalog phase; `source_execution_phase` diagnostic |
| **Harness** | Stage-15 restores `prompt_defense_enabled` product default |

### Scoped qualification (post-IMPL)

| Check | Result |
|-------|--------|
| UAEP typed payload @ `BEFORE_CONTEXT_BUILD` | **PASS** — `test_uaep_executor.py` |
| Prompt Defense allow/block on UAEP path | **PASS** — `test_uaep_executor.py`, `test_application_security_wiring.py` |
| Hook violation schema + diagnostics | **PASS** — `test_hook_runtime_guard.py` |
| Telemetry publish failure ≠ ALLOW | **PASS** — `test_hook_runtime_guard.py` |
| D3 recovery / restart / descriptor corruption | **PASS** — mandatory regression command |
| Canonical lifecycle + active Prompt Defense | **PASS** — `test_canonical_agent_lifecycle_reaches_serving_and_execution` |

### Unresolved (post-IMPL)

| Class | Item |
|-------|------|
| **TRACKED FREEZE DEBT** | Global **FRZ-*** and **PROD-Q-R1-R7** closure remain operator-gated |
| **INDEPENDENT AUDIT** | IMPL-01 **READY FOR AUDIT** — not CLOSED |

**Does not** promote D3 production security slice to accepted; **does not** infer production recovery acceptance from Stage 15 alone.
