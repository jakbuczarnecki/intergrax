# INTEGRAx-NPSC-5F-PROTECTED-DRIFT-CLASSIFICATION-AND-RESIGNOFF

## Metadata

| Field | Value |
| ----- | ----- |
| **Task** | `INTEGRAx-NPSC-5F-PROTECTED-DRIFT-CLASSIFICATION-AND-RESIGNOFF` |
| **Date** | 2026-09-13 |
| **Branch** | `development` |
| **Classifier HEAD (audit session)** | `6c25ee506949fc4c1f0739fecfaa9e095927563f` |
| **Platform assurance closure** | `5dc6d52ef8c79605083df8c1b404f12f4be000ab` |
| **Post-closure boundary/docs commit** | `2eed1cc5fe67a45d615b5485fe58b726e4398bdc` |
| **Production drift audit target** | `962bf1ade25b220873cb724523ffeff1abf0fbc7` |
| **NPSC-5F Evidence Plane baseline** | `7a3569c64e892588992635c9cee10c264a9fc200` |
| **Governance SSOT** | [`INTEGRAX_POST_FREEZE_EVOLUTION_GOVERNANCE.md`](INTEGRAX_POST_FREEZE_EVOLUTION_GOVERNANCE.md) |

**Status:** Classification complete — **Class C**; **no resignoff** of protected-drift fingerprints in this task.

---

## Audit Target

Independent review of production semantic drift introduced by:

```text
962bf1ade25b220873cb724523ffeff1abf0fbc7
refactor(observability): bind causal evidence to execution identity
```

Context window for inventory (not whole-repo audit):

```text
5dc6d52ef8c79605083df8c1b404f12f4be000ab .. 962bf1ade25b220873cb724523ffeff1abf0fbc7
```

Commits **after** `962bf1ade` on `development` at session time (`d37ebb796`, `6c25ee506`) are **out of scope** for attributing `962bf1ade` semantics (EE-B2 chaos certification only).

---

## Diff Inventory

### Separated: `2eed1cc5f` (not `962bf1ade`)

| File | Type | Layer | Semantic impact |
| ---- | ---- | ----- | --------------- |
| `docs/project/architecture/DIAGNOSTICS.md` | M | Docs | Ownership / boundary clarification |
| `docs/project/architecture/OBSERVABILITY.md` | M | Docs | Evidence vs diagnostics ownership |
| `docs/project/maintainers/plans/OBSERVABILITY.md` | M | Docs | Plan alignment |
| `intergrax/runtime/diagnostics/execution_reconstruction.py` | M | Runtime | **Docstring only** — no behavioral diff |

**Classification (isolated):** documentation / provenance clarification — **not** the production contract drift under review.

### Production: `962bf1ade` only

| File | Type | Layer | Semantic impact |
| ---- | ---- | ----- | --------------- |
| `intergrax/runtime/observability/causal_evidence.py` | M | Contract/Runtime | `platform_causal_evidence.v1` → **v2**; `RuntimeExecutionRef` + mandatory `ExecutionId` |
| `intergrax/runtime/observability/causal_evidence_export.py` | M | Contract/Runtime | Export mapping includes `target_execution_id` |
| `intergrax/runtime/observability/export_boundary.py` | M | Contract (R3 frozen) | `CausalEvidenceExportSource.target_execution_id` added |
| `intergrax/runtime/observability/persistence_conformance.py` | M | Test harness | Conformance samples require `execution_id` on causal targets |
| `intergrax/runtime/execution/identity_authority.py` | M | Runtime | `BackgroundTransportIdentity` + `execution_id`; bootstrap mint uses `mint_root_execution_identity()` |
| `intergrax/runtime/background_execution/bootstrap.py` | M | Runtime | `BackgroundExecutionIdentity.execution_id` propagated |
| `intergrax/runtime/background_execution/identity_persistence.py` | M | Persistence | Partition `intergrax.bg_exec_identity.v1` → **v2**; 4-field durable encoding |
| `intergrax/runtime/background_execution/reentry_admission.py` | M | Runtime | Identity field propagation |
| `intergrax/runtime/background_execution/required_audit_evidence.py` | M | Runtime | Causal evidence binds persisted execution identity |
| `docs/project/architecture/OBSERVABILITY.md` | M | Docs | OBS-CAUSAL-2 narrative |
| `docs/project/maintainers/plans/OBSERVABILITY.md` | M | Docs | OBS-CAUSAL-2 closure |
| `testing_support/...` + `tests/**` | M | Test | Contract alignment — not resignoff-qualified without Class A/B |

---

## Frozen/Public Surface Provenance

Certified NPSC-5F Evidence Plane surfaces (see [`EE_FINAL_02_NPSC5F_FROZEN_PLANE_DRIFT_RECONCILIATION_AND_ENTERPRISE_RE_FREEZE.md`](EE_FINAL_02_NPSC5F_FROZEN_PLANE_DRIFT_RECONCILIATION_AND_ENTERPRISE_RE_FREEZE.md), `NPSC_5F_EXECUTION_EVIDENCE_REPLAY_OBSERVABILITY_ARCHITECTURE.md`):

| Symbol / surface | Pre-`962bf1ade` certified? | Evidence |
| ---------------- | -------------------------- | -------- |
| `RuntimeExecutionRef` | **Yes** — `intergrax/runtime/observability/causal_evidence.py` @ closure | `platform_causal_evidence.v1`; fields `task_id`, `run_id`, `attempt_id`, `tenant_id` (no `execution_id`) |
| `PlatformCausalEvidence` | **Yes** — same module | Schema literal `platform_causal_evidence.v1` |
| `CausalEvidencePersistence` / persistence contracts | **Yes** — R1 qualified | Stores canonical `PlatformCausalEvidence` records |
| `export_boundary.CausalEvidenceExportSource` | **Yes** — R3 frozen exact path | R3 protected drift sentinel |
| Execution reconstruction | **Yes** — R4 read-only | Unchanged behavior in `962bf1ade` |
| Runtime event identity spine | **Yes** — five-ID `RuntimeEvent` | Pre-existed; causal ref lagged until `962bf1ade` |

Protected-drift classifiers (no guesswork):

- **NPSC-5F Final:** `testing_support/npsc5f_final_evidence_plane_drift.py` — `intergrax/runtime/observability/**` (except qualified prefixes) = **BREAKING** when changed since baseline `7a3569c64…`.
- **R3:** `export_boundary.py` frozen since `R3_IMPLEMENTATION_SHA`.
- **R1/R2/R4:** no file-level hit from `962bf1ade` alone (R2 journal paths unchanged in `5dc6d52..962bf1ade` diff).

---

## Identity Provenance

| ID | Canonical owner | Approved mint path (pre/post `962bf1ade`) | `962bf1ade` effect |
| -- | --------------- | ------------------------------------------- | ------------------ |
| `TaskId` | Execution identity authority | `mint_task_id()` / authority | Unchanged owner |
| `RunId` | `DefaultExecutionIdentityAuthority` | `mint_run_identity()` / `mint_root_execution_identity` | Bootstrap now ties run to root mint bundle |
| `AttemptId` | Same | `mint_attempt_identity()` / root bundle | Same |
| `ExecutionId` | Same | `mint_execution_identity()` / `mint_root_execution_identity` | **Now required** on transport bootstrap + causal target; not minted by Observability on admission path |

**`mint_background_transport_identity()` → `mint_root_execution_identity()`:** reuses existing execution authority (`identity_authority.py`); does **not** introduce a second mint owner. Changes **completeness** of transport-persisted identity (adds `execution_id` aligned with root mint), not a new authority domain.

**Observability minting:** `build_transport_triggered_execution_evidence` copies `execution_identity.execution_id` from `BackgroundExecutionIdentity`; production admission does not synthesize missing `ExecutionId`. Conformance harness `sample_causal_evidence` may call `mint_execution_id()` **only** in test/conformance helpers — not on the required-audit production builder.

---

## Evidence Contract Analysis

**Before @ `5dc6d52ef`:**

- `PLATFORM_CAUSAL_EVIDENCE_SCHEMA = "platform_causal_evidence.v1"`.
- `RuntimeExecutionRef`: `task_id`, `run_id`, `attempt_id`, `tenant_id` (`extra="forbid"`, frozen).

**After @ `962bf1ade`:**

- `platform_causal_evidence.v2`.
- `RuntimeExecutionRef` adds **required** `execution_id: ExecutionId`.

**Evidence ≠ control:** unchanged — `PlatformCausalEvidence` still records transport→execution **relation**; no scheduler/retry/policy steering added in this commit.

**Dependency direction:** Execution identity (bootstrap / authority) → persisted identity → `PlatformCausalEvidence.target`; evidence does not create missing execution identity on the required-audit path.

---

## Persistence Schema Analysis

| Store | Before | After | v1 read |
| ----- | ------ | ----- | ------- |
| Causal evidence canonical record | v1 schema in model | v2 only | **No** v1 deserializer/migration in tree — v1 payloads incompatible with v2 model |
| KV/document background identity | 3-field record / `intergrax.bg_exec_identity.v1` | 4-field / **v2** partition | **Fail-closed** — `_decode_identity_record` requires `len(parts) == 4` |
| Export envelope field set | No `target_execution_id` | Required `target_execution_id` | Breaking for consumers of `CausalEvidenceExportSource` |

**Persistence contract semantics:** **changed** (durable encoding and partition version). No documented migration path for in-flight v1 background identity or v1 causal evidence blobs.

---

## Compatibility Matrix

| Surface | Before | After | Breaking? | Frozen? | Class impact |
| ------- | ------ | ----- | --------: | -------: | ------------ |
| `RuntimeExecutionRef` | No `ExecutionId` | Mandatory `ExecutionId` | **Yes** | Yes (OBS / NPSC-5F) | **C** |
| `PlatformCausalEvidence` schema | v1 | v2 | **Yes** | Yes | **C** |
| `BackgroundExecutionIdentity` | 3 IDs | + `execution_id` | **Yes** | Extension surface | **C** (durable semantics) |
| `PersistedBackgroundExecutionIdentity` | 3-part encoding | 4-part + v2 partition | **Yes** | Durable | **C** |
| `CausalEvidenceExportSource` | 3 target IDs | + `target_execution_id` | **Yes** | R3 frozen | **C** |
| Background bootstrap mint | run/attempt minted separately | `mint_root_execution_identity()` bundle | Compatible authority | Execution | **B** at most — overshadowed by contract breaks |
| `RuntimeEvent` five-ID spine | Present | Unchanged in `962bf1ade` | No | Yes | — |

---

## Protected Drift Sentinel Analysis

Drift detected from baseline `7a3569c64…` → `origin/development` (includes `962bf1ade`; unchanged by later EE-B2-only commits for FINAL/R3 lists).

| Test | Protected surface | Why it failed | Legitimate drift? | Resignoff possible? |
| ---- | ----------------- | ------------- | ----------------- | ------------------- |
| `test_final_no_breaking_protected_drift_since_baseline` | NPSC-5F Final Evidence Plane (`causal_evidence*.py`, `export_boundary.py`, `persistence_conformance.py`) | `collect_breaking_evidence_plane_production_drift` non-empty | **Yes** — semantic contract edits without qualified baseline advance | **No** without Class C reopen + re-qualification |
| `test_npsc5f_final_predecessor_drift_sentinels_empty` | Aggregates R1/R2/R3 + Final | Fails on R3 + Final (R1/R2 empty for `962` range) | **Yes** | **No** |
| `test_r3_final_no_unqualified_protected_drift_since_implementation` | `intergrax/runtime/observability/export_boundary.py` | `export_boundary.py` in R3 drift set | **Yes** — new export field | **No** without scoped R3 re-freeze |

**Not updated in this task:** sentinel SHAs, digest hashes, or protected file lists (per fail-safe — updating before classification would bypass guards).

Functional note: R2 sentinel remains **empty** for `5dc6d52..962bf1ade`; the “3 failed” targeted assurance count matches Final + R3 sentinels (duplicate Final assertion in `test_npsc5f_final_protected_drift` is the same BREAKING set).

---

## A/B/C Classification

| Candidate | Verdict | Rationale |
| --------- | ------- | --------- |
| Class A | **Rejected** | Existing public contracts changed (`v1`→`v2`, mandatory field, R3 export shape) |
| Class B | **Rejected** | Not implementation-only; public serialization and persistence semantics changed |
| Class C | **Selected** | Governance automatic triggers: existing frozen public contract change; persistence contract semantic change; NPSC-5F protected-plane BREAKING drift |

**Formal decision:** **CLASS C REQUIRED**

---

## Architecture Reopen Decision

```text
Architecture Reopen required: YES
```

Scoped reopen record required before:

- advancing NPSC-5F Final / R3 qualified baselines,
- updating protected-drift fingerprints,
- treating OBS-CAUSAL-2 as post-freeze certified closure.

**Not in scope of this task:** migration design, production refactors, sentinel hash updates.

---

## Test Evidence

Session commands (logs under `.tmp/session/NPSC-5F-PROTECTED-DRIFT/`):

| Suite | Result | Notes |
| ----- | ------ | ----- |
| `test_final_no_breaking_protected_drift_since_baseline` | **FAIL** | 4 BREAKING paths |
| `test_r3_final_no_unqualified_protected_drift_since_implementation` | **FAIL** | `export_boundary.py` |
| `test_npsc5f_final_predecessor_drift_sentinels_empty` | **FAIL** | Same drift aggregation |
| Causal + background identity unit tests (`test_causal_evidence_*`, `test_background_execution_*`) | **PASS** (in targeted run) | Functional alignment with v2 — does not negate Class C |
| `test_execution_identity_single_authority_gate` | **PASS** | |
| `test_npsc5f_p0_execution_evidence_architecture_reconciliation` | **PASS** | |
| `test_npsc5f_r1_durable_evidence_persistence_boundary_resignoff` | **PASS** | R1 paths untouched by `962` |
| `test_execution_identity.py::test_unified_task_runner_mints_attempt_at_run_boundary` | **FAIL** | `AttributeError: '_StubLoop' object has no attribute 'event_bus'` — appears unrelated to `962bf1ade`; not used to downgrade Class C |

**Resignoff gates (Class A/B):** **not executed** — Class C stop rule.

---

## Static Quality

`uv run ruff check` on core `962bf1ade` production files @ session HEAD:

- `F401` unused import `mint_background_transport_identity` in `identity_persistence.py`
- Pre-existing `E402` late imports in `export_boundary.py`
- `ruff format --check` would reformat several touched modules

No pyright run for this classification-only deliverable (no production edits authorized).

---

## Findings

### Critical

- None — Observability does **not** mint `ExecutionId` on the required-audit evidence builder; no second identity authority introduced.

### Major

- Breaking change to certified causal evidence public contract (`v1` → `v2`, mandatory `ExecutionId`).
- Breaking R3 export surface (`target_execution_id`).
- Durable background identity v1 records incompatible (fail-closed) without migration policy.
- Protected-drift sentinels correctly fail — drift is **legitimate** and **unqualified**.

### Minor

- Post-`962` architecture docs describe OBS-CAUSAL-2 closure while NPSC-5F fingerprints still pin pre-drift baseline.
- Ruff F401 on unused import in identity persistence (hygiene).

---

## Final Verdict

```text
NPSC-5F PROTECTED DRIFT REQUIRES SCOPED ARCHITECTURE REOPEN
```

**Production changes in this task:** **NO**  
**Protected-drift fingerprint updates:** **NO**  
**Parallel EE-B2 WIP:** clean working tree at session start — **not touched**

---

Wprowadzone zmiany wymagają audytu na podstawie aktualnego kodu znajdującego się na GitHub. Raport implementacji nie zastępuje niezależnej weryfikacji zmian wykonanych w repozytorium.
