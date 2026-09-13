# INTEGRAx-POST-FREEZE-ARCHITECTURE-GUARD-MATRIX

## Metadata

| Field | Value |
| ----- | ----- |
| **Task** | `INTEGRAx-POST-FREEZE-ARCHITECTURE-GUARD-MATRIX` |
| **Date** | 2026-09-13 |
| **Branch policy** | `development` |
| **Frozen code baseline SHA** | `a185403d0c7524c29bea2fe09212f9508e6bccd8` |
| **Freeze record commit SHA** | `59fbf6f305b70d2b74adac7cd61dd21d352dba78` |
| **Freeze provenance correction SHA** | `e60fc0162e3302d5be0c320593755a9657de23ce` |
| **Post-freeze governance SHA** | `3014bd300947920febc0eeae568e58c177eed800` |
| **Frozen extension-point certification SHA** | `a99cee7dea631bd9080c238d3a06887c2e72f880` |
| **Parent governance SSOT** | [`INTEGRAX_POST_FREEZE_EVOLUTION_GOVERNANCE.md`](INTEGRAX_POST_FREEZE_EVOLUTION_GOVERNANCE.md) |
| **Parent freeze SSOT** | [`INTEGRAX_CORE_PLATFORM_FREEZE.md`](INTEGRAX_CORE_PLATFORM_FREEZE.md) |
| **Extension surfaces SSOT** | [`INTEGRAX_FROZEN_EXTENSION_POINT_CERTIFICATION.md`](INTEGRAX_FROZEN_EXTENSION_POINT_CERTIFICATION.md) |

**Status:** **Post-freeze architecture guard matrix = ACTIVE**

**Production modifications in this task:** **NONE**

This document is the **single maintainer SSOT** mapping **guard families** → **frozen invariants** → **existing representative gates** → **change class / archetype requirements**. It does **not** duplicate CI orchestration, add new policy engines, or reopen frozen core.

---

## Scope

**In scope:** Architecture assurance for post-freeze changes (Class A / B / C): which guards apply, which gates must run (MANDATORY / CONDITIONAL / N/A), severity on failure, reopen triggers, and gate dependencies.

**Out of scope (parallel sessions — do not modify or certify here):** EE-B2 chaos engineering, EE-B3 security hardening, EE-B4 operational excellence, and any WIP untracked artifacts for those tracks.

**Enforcement surface:** This SSOT + **existing** architecture / contract tests referenced below. **Reuse existing gates first** — new tests only when a real guard has no representative gate and can be added without production change (separate task).

---

## Guard Philosophy

Every post-freeze change must answer:

| # | Question |
| - | -------- |
| 1 | **What changed?** (surfaces, packages, contracts) |
| 2 | **Which frozen surface could be affected?** (Frozen Surface Registry / INV / REL) |
| 3 | **Which gates must run?** (mandatory vs optional) |
| 4 | **Does this require independent audit?** (per class) |
| 5 | **Does this require Architecture Reopen?** (Class C) |

**Fail-safe:** If a guard failure can breach a frozen invariant, or classification is ambiguous → **escalate to Class C** until proven otherwise.

**Class A minimum:** Class A is **not** “run nothing because extension is additive.” Minimum: extension contract tests, provider/plugin tests where applicable, **impacted architecture guards**, targeted regression, static quality in scope.

---

## Guard Family Matrix

| Guard Family | Frozen Invariant | Representative Existing Gate | Required For | Severity on Failure | Reopen Trigger? |
| ------------ | ---------------- | ---------------------------- | ------------ | ------------------- | --------------- |
| **1. Execution ownership** | Single `ExecutionRuntime` ownership; single root lifecycle; no second scheduler / execution loop | `tests/unit/runtime/architecture/test_ee_a1_execution_engine_ownership_certification_gate.py` | Any frozen execution / scheduler / lifecycle touch; Class B bugfix on runtime | **Critical** | YES if ownership moves or second loop appears |
| **2. Canonical path / zero-bypass** | Decision → Governance → `DecisionExecutionAuthorization` → `ExecutionRequest` → `ExecutionRuntime` | `tests/unit/runtime/architecture/test_platform_execution_unification_u5_final_zero_bypass.py` | Execution dispatch, admission, plugin/tool paths that can run work | **Critical** | YES on alternate canonical path |
| **3. Governance authorization boundary** | `DENY` / `REQUIRE_HUMAN` / `ALLOW`; authorization minting boundary | `tests/unit/contracts/test_decision_contract_architecture_gates.py`, `tests/unit/runtime/architecture/test_npsc42_h1_governance_boundary_freeze.py` | Governance, decision authorization, policy plugins | **Critical** | YES on semantic change |
| **4. Retry ownership** | Single qualified retry owner; attempt semantics; no retry engine in plugin/provider | `tests/unit/runtime/architecture/test_npsc5e_r1_execution_retry_attempt_semantics.py`, `tests/unit/runtime/architecture/test_npsc5e_r1_final_retry_attempt_qualification.py` | Retry policy, worker redelivery, background identity | **Major** (Critical if second owner) | **YES** on retry ownership change |
| **5. Recovery ownership** | Recovery Plane ownership; no recovery executors in plugins/providers; no alternate recovery path | `tests/unit/runtime/architecture/test_npsc5e_final_recovery_plane_qualification_and_freeze.py`, `tests/unit/runtime/architecture/test_w3_c4_decision_durable_recovery_canonical_wiring.py` | Recovery, checkpoint resume, durable decision recovery | **Major** (Critical if bypass) | YES on recovery authority change |
| **6. Persistence abstraction** | Engine/Core → persistence **contract** → provider → vendor (not direct DB/vendor in core) | `tests/unit/runtime/architecture/test_npsc5f_r1_durable_evidence_persistence_boundary_resignoff.py`, `tests/unit/runtime/architecture/test_ee_b1_1_persistence_failure_contract.py` | Persistence providers, event/evidence stores, fail-closed persistence | **Major** (Critical if authority moves) | YES on contract/authority semantic change |
| **7. Evidence ≠ control** | Evidence = truth/record; evidence must not drive execution control | `tests/unit/runtime/architecture/test_npsc5f_p0_execution_evidence_architecture_reconciliation.py`, `tests/unit/runtime/architecture/test_npsc5f_final_evidence_plane_qualification.py` | Evidence plane, export, reconstruction, audit admission | **Critical** | YES if evidence controls execution |
| **8. Identity authority** | No minting of `run_id`, `execution_id`, `attempt_id`, `task_id` outside approved authority | `tests/unit/runtime/architecture/test_execution_identity_single_authority_gate.py`, `tests/unit/runtime/architecture/test_ee_a2_identity_authority_certification.py` | Identity, lineage, background redelivery, nexus handles | **Critical** | YES on authority change |
| **9. Tracing public contracts** | Typed public trace values; vendor-neutral; bounded serialization | `tests/unit/contracts/test_tracing_public_contract.py` | Public trace types, export envelopes, cross-tier contracts | **Major** | YES on breaking public trace contract |
| **10. Reliability contracts** | EE-B1.1 failure classification; fail-closed persistence; shutdown semantics | `tests/unit/runtime/architecture/test_ee_b1_1_failure_semantics_certification.py`, `tests/unit/runtime/architecture/test_ee_b1_1_shutdown_contract.py` | Reliability, worker failure, persistence failure handling | **Major** (Critical if fail-open on invariant path) | YES on REL semantic change |
| **11. Plugin boundaries** | No concrete plugin imports in core; manifest/admission; no plugin-owned lifecycle / hidden dispatch | `tests/unit/runtime/architecture/test_ds_plugin_architecture_gates.py`, `tests/unit/runtime/architecture/test_hardening_5_plugin_architecture_gate.py`, `tests/unit/runtime/architecture/test_plugin_ep_scanner_consolidation_gate.py` | New/changed plugins, admission, composition wiring | **Major** (Critical if bypass) | YES if plugin owns lifecycle or bypasses governance |
| **12. Provider / vendor neutrality** | Core depends on contract; vendor logic in provider; no `if vendor ==` in frozen core | DS-PLUGIN / HARDENING-5 gates + extension cert grep policy; `tests/unit/runtime/execution/test_inference_profile_resolution.py` (profile resolution slice) | New providers, adapters, LLM/storage integrations | **Major** | YES if core requires vendor branch |
| **13. Composition-root ownership** | Composition binds; does not redefine semantics; no local governance; no alternate runtime | `tests/unit/runtime/architecture/test_npsc4_2_residual_compatibility_gate.py`, `tests/unit/applications/architecture/test_binding_contract_identity_authority.py` | Hosted profiles, app composition, production wiring | **Major** (Critical if alternate runtime) | YES on production composition ownership change |
| **14. Nexus ownership** | Orchestration ownership; no duplicate fan-out/graph lifecycle | `tests/unit/runtime/architecture/test_npsc4_2_residual_compatibility_gate.py`, `tests/unit/runtime/architecture/test_npsc4_agent_runtime_governance_gate.py` | Nexus graph, fan-out, multi-agent orchestration | **Critical** if duplicate lifecycle | YES on Nexus ownership invariant change |
| **15. Child execution ownership** | Canonical child path; no second root capacity ownership | `tests/unit/runtime/architecture/test_platform_execution_unification_u4_child_execution_closure.py`, `tests/unit/runtime/architecture/test_ee_b1_2_capacity_child_execution_interaction.py` | Child runs, partial fan-out, nested execution | **Major** | YES if child path bypasses canonical runner |
| **16. Capacity / backpressure ownership** | Capacity preview ≠ admission ownership ≠ execution ownership (EE-B1.2) | `tests/unit/runtime/architecture/test_ee_b1_2_capacity_architecture_gate.py`, `tests/unit/runtime/architecture/test_ee_b1_2_capacity_admission.py` | Capacity evaluators, admission, concurrency limits | **Major** | YES if capacity layer owns execution |
| **17. Diagnostic extension isolation** | Extension isolated; no persistence authority; no runtime ownership | `tests/unit/contracts/test_diagnostic_extension_spi.py`, `tests/unit/runtime/diagnostics/test_diagnostic_extension_spi_r5_qualification.py` | Optional diagnostic extensions | **Major** | YES if extension gains control plane |
| **18. Inference / model provider abstraction** | Logical profile → resolver/catalog → adapter → vendor; no vendor branch in core | `tests/unit/runtime/execution/test_inference_profile_resolution.py`, host composition binding tests per extension cert | New models, adapters, profile catalogs | **Major** | YES if core branches on vendor |

**Gate applicability legend (typical):**

| Label | Meaning |
| ----- | ------- |
| **MANDATORY** | Must run before merge when change touches the family or archetype default requires it |
| **CONDITIONAL** | Run when diff touches paths named in family scope or integration can affect invariant |
| **NOT APPLICABLE** | Safe to skip when change is provably outside family (document in PR / audit) |

---

## Class A/B/C Gate Matrix

| Change Class | Required Guard Families | Independent Audit | Architecture Reopen |
| ------------ | ----------------------- | ----------------: | ------------------: |
| **A** — Safe extension | **Impacted** extension families + targeted regression + static quality; **never** “no guards because additive” | Risk-based; **required** where governance defines (security, persistence, auth); else recommended for high-risk extensions | **NO** |
| **B** — Core-compatible | **All Class A minimum** + **impacted frozen core families** (from matrix) + static quality | **YES** (GitHub code, not report alone) | **NO** (misclassification → treat as **C**) |
| **C** — Contract / ownership evolution | **Architecture Reopen Record** + **all impacted families** + migration + scoped/full recertification | **YES** | **YES**; explicit re-freeze only if new baseline desired |

**Mandatory gate bundles (minimum, add impacted rows from Guard Family Matrix):**

| Class | Minimum bundle |
| ----- | ---------------- |
| **A** | Contract/plugin/provider tests for the extension + static quality + **CONDITIONAL→MANDATORY** for any family the change could affect |
| **B** | Class A + EE-A1 + U5 when execution touched + reliability/capacity/retry/recovery/governance/persistence/evidence/identity rows as impacted + independent audit |
| **C** | Reopen doc + full impacted family gates + recertification scope from reopen record + audit + re-freeze record only after explicit baseline selection |

---

## Change Archetype Matrix

| Change Archetype | Class Default | Mandatory Guards (representative gates) |
| ---------------- | ------------- | ---------------------------------------- |
| **New plugin** | **A** | Plugin (DS-PLUGIN, HARDENING-5, EP scanner); **CONDITIONAL→MANDATORY** U5 if plugin can request execution; governance gates if policy surface; static + targeted regression |
| **New provider (generic port)** | **A** | Provider neutrality + contract tests; persistence/tracing/inference family as applicable; static |
| **New persistence provider** | **A** | Persistence abstraction + `test_ee_b1_1_persistence_failure_contract.py` + NPSC-5F persistence boundary; reliability fail-closed; static; audit **recommended** |
| **Execution policy / evaluator** | **A** | EE-A1 (no lifecycle ownership); capacity family if admission-related; static |
| **Internal ExecutionRuntime bugfix** | **B** | EE-A1, U5, impacted EE-B1.1 / B1.2 / retry / recovery / identity; targeted regression; **independent audit** |
| **Public contract semantic change** | **C** | Architecture Reopen + all impacted families + migration + recertification |
| **Governance change** | **C** (default) | Governance + U5 + zero-bypass; reopen unless proven behavior-preserving **B** with audit |
| **Observability / evidence exporter** | **A** | Evidence ≠ control (NPSC-5F); tracing public contract if public types change; static |
| **Inference adapter / model provider** | **A** | Inference abstraction + provider neutrality; static; no core vendor branches |
| **Composition change** | **A** if new allowed binding; **C** if ownership semantics change | Nexus/composition (NPSC-4.2), binding identity gates; U5 if execution wiring changes |

---

## Ownership Guards

**Execution (EE-A1):** Protects single runtime owner, root lifecycle, prohibition of second scheduler. Gate: `test_ee_a1_execution_engine_ownership_certification_gate.py`.

**Zero-bypass (U5):** Protects canonical Decision → Execution path. Gate: `test_platform_execution_unification_u5_final_zero_bypass.py`.

**Child execution:** Protects canonical child runner; interacts with capacity without owning execution. Gates: U4 child closure, EE-B1.2 child/capacity interaction tests.

**Nexus:** Protects orchestration ownership. Gates: `test_npsc4_2_residual_compatibility_gate.py`, agent runtime governance gate.

**Capacity (EE-B1.2):** Preview/admission must not become execution owner. Gates: `test_ee_b1_2_capacity_architecture_gate.py` and related EE-B1.2 suite.

---

## Persistence Guards

**Invariant:** INV-7 / persistence freeze rule — `core → contract → provider → vendor`.

**Gates:** `test_npsc5f_r1_durable_evidence_persistence_boundary_resignoff.py`, `test_ee_b1_1_persistence_failure_contract.py`, plus EE-B1.1 reliability persistence failure contract.

**Change retry/recovery persistence coupling:** Treat as **impacted** reliability + recovery families; ownership shift → **Class C**.

---

## Plugin / Provider Guards

**Plugins:** Manifest/admission, dependency direction, no core → concrete plugin orchestration. Gates: DS-PLUGIN, HARDENING-5, EP scanner; integration reference `tests/integration/platform_plugins/test_plugin8_dual_mode_tool_e2e.py` for representative behavior (Class A regression, not a substitute for architecture gates).

**Providers:** Port injection, vendor logic out of core. Architecture enforcement via plugin/provider gate families + contract tests; **forbidden** `if vendor ==` / `if plugin ==` in frozen core (see governance compatibility rules).

---

## Execution Guards

Combined **ownership + zero-bypass + child + capacity** for any change that can schedule, admit, or run work. Class **B** runtime bugfixes: always include EE-A1 + U5 at minimum, then expand using diff scope.

**Legacy path retirement (conditional):** `tests/unit/runtime/architecture/test_ue_9d_legacy_execution_retirement_gate.py` when touching retirement/routing surfaces listed in freeze SSOT.

---

## Governance Guards

Protect `DENY` / `REQUIRE_HUMAN` / `ALLOW` and authorization minting. Primary gates: `test_decision_contract_architecture_gates.py`, `test_npsc42_h1_governance_boundary_freeze.py`. Any semantic governance change defaults **Class C**.

---

## Reliability Guards

EE-B1.1 certification and related shutdown / persistence failure contracts. Primary: `test_ee_b1_1_failure_semantics_certification.py`. Reliability must not take persistence **ownership** (REL-4) — pair with persistence guard when diff spans both.

---

## Evidence Guards

NPSC-5F reconciliation and evidence plane qualification — evidence records truth, must not control execution. Primary: `test_npsc5f_p0_execution_evidence_architecture_reconciliation.py`, `test_npsc5f_final_evidence_plane_qualification.py`.

---

## Identity Guards

Single authority for execution identifiers. Gates: `test_execution_identity_single_authority_gate.py`, EE-A2 identity certification family, `tests/unit/contracts/test_execution_identity.py` for contract slice.

---

## Composition Guards

Composition binds implementations; must not introduce alternate runtime or local governance. Gates: NPSC-4.2 residual compatibility, application binding identity architecture tests. Production composition **ownership** change → **Class C**; new conforming binding → **Class A** with impacted guards.

---

## Gate Dependencies

Explicit ordering for review (not a graph engine):

```text
Execution ownership (EE-A1)
  → Zero-bypass (U5)
  → Governance authorization (when policy can admit execution)
  → Child execution
  → Capacity / backpressure (EE-B1.2)
Retry ownership → Recovery ownership (recovery must not re-home retry)
Persistence abstraction → Reliability fail-closed (EE-B1.1)
Evidence ≠ control → Observability export (export must not become control plane)
Identity authority → Tracing public contracts (when IDs appear in public trace)
Plugin / provider boundaries → Composition-root wiring
Nexus ownership → Child execution / fan-out
```

When a downstream guard fails, assume upstream invariants may be violated until proven otherwise → **fail-safe to Class C**.

---

## Severity Mapping

| Severity | Guard failure examples |
| -------- | ---------------------- |
| **Critical** | Governance bypass; second execution runtime / scheduler; evidence controls execution; unauthorized identity mint; alternate canonical execution path |
| **Major** | Direct vendor coupling in frozen core; duplicate retry/recovery owner; provider requires core branch; plugin owns lifecycle; missing mandatory Class B gate |
| **Minor** | Missing doc cross-link; ambiguous gate scope in PR; incomplete certification traceability |

---

## Architecture Reopen Triggers

**Always Class C (non-exhaustive — see governance SSOT):**

- Public contract semantic or breaking change
- Execution / lifecycle / retry / recovery **ownership** change
- Governance or `DecisionExecutionAuthorization` semantic change
- Persistence **contract authority** change
- Identity minting authority change
- Canonical path alteration or bypass
- Any frozen **INV-1 … INV-10** or **REL-1 … REL-8** semantic change
- Nexus / production composition **ownership** change (not a new allowed Class A binding)

**Reopen record path:** `docs/project/maintainers/qualification/ARCHITECTURE_REOPEN_<TASK>.md` per governance SSOT.

**Re-freeze:** Only after Class C recertification + explicit baseline selection — never automatic on merge.

---

## Findings

| Severity | Finding |
| -------- | ------- |
| **Observation** | Parallel EE-B2 chaos WIP exists as **untracked** files only — out of scope; guard matrix does not reference EE-B2 gates for mandatory post-freeze regression. |
| **Observation** | Governance SSOT already lists a **frozen gate families (reference)** table; this matrix **extends** it with severity, reopen triggers, archetypes, MANDATORY/CONDITIONAL/N/A, and dependencies without changing A/B/C semantics. |
| **Minor** | Persistence family has multiple representative gates across NPSC-5F and EE-B1.1 — operators must select **impacted** gates from the matrix, not run the entire NPSC-5F suite for every Class A change. |

No **Critical** or **Major** architecture defects identified in production code during this documentation task.

---

## Final Verdict

```text
POST-FREEZE ARCHITECTURE GUARD MATRIX ESTABLISHED
```

---

## Representative gate validation (task evidence)

Commands used when establishing this SSOT (single pytest invocation — not full repo suite):

```text
uv run pytest \
  tests/unit/runtime/architecture/test_ee_a1_execution_engine_ownership_certification_gate.py \
  tests/unit/runtime/architecture/test_platform_execution_unification_u5_final_zero_bypass.py \
  tests/unit/runtime/architecture/test_ee_b1_1_failure_semantics_certification.py \
  tests/unit/runtime/architecture/test_ee_b1_2_capacity_architecture_gate.py \
  tests/unit/runtime/architecture/test_ds_plugin_architecture_gates.py \
  tests/unit/runtime/architecture/test_hardening_5_plugin_architecture_gate.py \
  tests/unit/runtime/architecture/test_npsc5f_p0_execution_evidence_architecture_reconciliation.py \
  tests/unit/runtime/architecture/test_npsc5f_r1_durable_evidence_persistence_boundary_resignoff.py \
  tests/unit/contracts/test_diagnostic_extension_spi.py \
  -q
```

Docs-only static check: `git diff --check`

**EE-B2 chaos gates:** intentionally **excluded** (parallel WIP).
