# INTEGRAx-NPSC-5F-SCOPED-ARCHITECTURE-REOPEN

## Metadata

| Field | Value |
| ----- | ----- |
| **Task** | `INTEGRAx-NPSC-5F-SCOPED-ARCHITECTURE-REOPEN` |
| **Date** | 2026-09-13 |
| **Branch** | `development` |
| **Reopen record HEAD** | (set at commit — see git log) |
| **Global frozen code baseline** | `a185403d0c7524c29bea2fe09212f9508e6bccd8` — **unchanged** |
| **Platform assurance closure** | `5dc6d52ef8c79605083df8c1b404f12f4be000ab` |
| **Protected drift (semantic source)** | `962bf1ade25b220873cb724523ffeff1abf0fbc7` |
| **Classification SSOT** | `1858ba068d0ba652c6eba0ee875eda06a8dc6183` — **Class C** |
| **NPSC-5F Evidence Plane qualified baseline (pre-drift)** | `7a3569c64e892588992635c9cee10c264a9fc200` |
| **Prior classification** | [`INTEGRAX_NPSC_5F_PROTECTED_DRIFT_CLASSIFICATION_AND_RESIGNOFF.md`](INTEGRAX_NPSC_5F_PROTECTED_DRIFT_CLASSIFICATION_AND_RESIGNOFF.md) |
| **Governance SSOT** | [`INTEGRAX_POST_FREEZE_EVOLUTION_GOVERNANCE.md`](INTEGRAX_POST_FREEZE_EVOLUTION_GOVERNANCE.md) |

**Status:** Scoped architecture reopen — **approved**; v1/v2 compatibility implementation — **completed** (see [`INTEGRAX_NPSC_5F_V1_V2_COMPATIBILITY_AND_MIGRATION_IMPLEMENTATION.md`](INTEGRAX_NPSC_5F_V1_V2_COMPATIBILITY_AND_MIGRATION_IMPLEMENTATION.md)).

**Reopen type:**

```text
SCOPED ARCHITECTURE REOPEN
```

**Not:**

```text
FULL PLATFORM REOPEN
```

---

## Reopen Reason

Post-freeze production commit `962bf1ade25b220873cb724523ffeff1abf0fbc7` (`refactor(observability): bind causal evidence to execution identity`) changed certified NPSC-5F public and durable contracts without advancing qualified baselines or protected-drift fingerprints. Independent classification at `1858ba068d0ba652c6eba0ee875eda06a8dc6183` concluded **Class C** and **Architecture Reopen required: YES**.

This record formally reopens **only** the NPSC-5F surfaces touched by that drift so that v2 semantics, v1 compatibility, and migration can be designed and implemented in a follow-on task, then requalified and re-frozen for **NPSC-5F R3** and **NPSC-5F Final** — without moving the global frozen baseline `a185403d…`.

---

## Protected Drift Provenance

| Item | Value |
| ---- | ----- |
| Drift commit | `962bf1ade25b220873cb724523ffeff1abf0fbc7` |
| Attribution window | `5dc6d52ef8c79605083df8c1b404f12f4be000ab` .. `962bf1ade` (production semantics) |
| Sentinel expectation | NPSC-5F protected drift remains **RED** until implementation + requalification + resignoff |
| Fingerprint policy in this task | **No updates** |

Production files attributed to `962bf1ade` (inventory from classification):

- `intergrax/runtime/observability/causal_evidence.py`
- `intergrax/runtime/observability/causal_evidence_export.py`
- `intergrax/runtime/observability/export_boundary.py`
- `intergrax/runtime/observability/persistence_conformance.py` (conformance harness)
- `intergrax/runtime/execution/identity_authority.py` (transport identity completeness)
- `intergrax/runtime/background_execution/bootstrap.py`
- `intergrax/runtime/background_execution/identity_persistence.py`
- `intergrax/runtime/background_execution/reentry_admission.py`
- `intergrax/runtime/background_execution/required_audit_evidence.py`

Docs-only OBS-CAUSAL-2 narrative in the same commit window is **provenance**, not a substitute for NPSC-5F re-freeze.

---

## Scoped Surfaces

Reopen is limited to:

1. `RuntimeExecutionRef`
2. `PlatformCausalEvidence` / `PLATFORM_CAUSAL_EVIDENCE_SCHEMA`
3. `platform_causal_evidence.v1` → `platform_causal_evidence.v2`
4. `CausalEvidenceExportSource` (R3 export boundary)
5. Background execution identity persistence:
   - 3-ID → 4-ID durable model (`BackgroundExecutionIdentity` / `PersistedBackgroundExecutionIdentity`)
   - `intergrax.bg_exec_identity.v1` → `intergrax.bg_exec_identity.v2`
6. Propagation of canonical `ExecutionId` on transport → evidence → export paths (completeness, not new authority)
7. Compatibility / migration semantics for historical v1 artifacts
8. NPSC-5F **R3** protected surface (export)
9. NPSC-5F **Final** protected surface (evidence plane)

---

## Out-of-Scope Frozen Surfaces

Absolutely **not** reopened:

| Area | Status |
| ---- | ------ |
| EE-B2 / EE-B3 / EE-B4 | Out of scope |
| `ExecutionRuntime` ownership | Frozen |
| Governance semantics | Frozen |
| Retry owner | Frozen |
| Recovery owner | Frozen |
| Nexus owner | Frozen |
| Plugin framework | Frozen |
| Full persistence architecture | Frozen (only bg identity + causal evidence adapters in scope) |
| Global identity authority redesign | Frozen |
| Full platform re-freeze | **Prohibited** |
| `RuntimeEvent` five-ID spine (unchanged by drift) | Not reopened |
| R1/R2/R4 qualified paths except where v2 read/write touches shared stores | Not reopened unless implementation proves coupling |

---

## Old Contract Semantics

### Frozen surface delta (pre-`962bf1ade` @ assurance closure)

| Frozen Surface | Old Semantics | Proposed Semantics | Ownership Change? | Reopen Scope |
| -------------- | ------------- | ------------------ | ----------------: | ------------ |
| `RuntimeExecutionRef` | `task_id`, `run_id`, `attempt_id`, `tenant_id`; no `execution_id`; schema v1 | Same fields + **required** `execution_id: ExecutionId`; carried only on v2 records | **No** — field completeness on existing ref | Yes |
| `PlatformCausalEvidence` | `platform_causal_evidence.v1` | `platform_causal_evidence.v2` with v2 `RuntimeExecutionRef` | **No** | Yes |
| Causal evidence schema literal | Write/read model assumes v1 | v2 is canonical write; v1 legacy read policy (below) | **No** | Yes |
| Export boundary | `CausalEvidenceExportSource`: task/run/attempt targets | Adds **required** `target_execution_id` in v2 export contract | **No** | Yes (R3) |
| Background identity record | 3-part durable encoding (task, run, attempt) | 4-part encoding includes `execution_id` | **No** — persistence of authority-minted ID | Yes |
| Background identity partition/version | `intergrax.bg_exec_identity.v1` | `intergrax.bg_exec_identity.v2` | **No** | Yes |
| Execution identity propagation | Causal target could omit `ExecutionId` while runtime spine had five IDs | Bootstrap and required-audit evidence bind canonical `execution_id` end-to-end | **No** — propagation only | Yes |

---

## New Proposed Contract Semantics

### Formal question: what must change vs stay frozen?

**Must change (conscious, versioned):**

- Public schema version for platform causal evidence: **v1 remains semantically frozen**; **v2** is the new explicit version.
- `RuntimeExecutionRef.execution_id`: **REQUIRED** on all v2 evidence and v2 export targets.
- Durable background identity: **v2 partition** and **4-field** encoding as the only **write** format after migration implementation.
- `CausalEvidenceExportSource`: v2 export shape includes `target_execution_id`; exports must be **versioned**, not silently mutated.

**Must remain unchanged:**

- Single execution identity authority (`DefaultExecutionIdentityAuthority` / `mint_*` family).
- `ExecutionRuntime` as canonical execution owner.
- Governance, retry, recovery, Nexus ownership.
- Evidence records truth; evidence does **not** steer execution.
- Observability does **not** mint `ExecutionId` on production admission / required-audit paths.
- Global frozen baseline `a185403d0c7524c29bea2fe09212f9508e6bccd8`.

### Identity semantics (TaskId, RunId, AttemptId, ExecutionId)

The four-ID reference model means **more complete identity reference** for correlation across transport, persistence, evidence, and export. It is **not** a new identity authority and **not** a second mint domain.

### `RuntimeExecutionRef` — formal decision: `execution_id: REQUIRED` on v2

**Decision:** **YES** — on `platform_causal_evidence.v2`, `execution_id` is **required**.

| Aspect | Policy |
| ------ | ------ |
| Invariant | Every v2 causal evidence record and v2 export target must reference the **same** `ExecutionId` already minted by execution identity authority at run/attempt boundary (or root bundle for background transport), never synthesized by observability. |
| Compatibility impact | v1 records without `execution_id` are **incomplete** under v2; readers must not promote them to v2 without canonical enrichment or explicit migration. |
| Migration expectations | Implementation task: dual-read v1+v2; write v2 only; enrichment/migration only from canonical sources (see below). |
| Consumer behavior | Consumers requiring full execution correlation **must** read v2 or enriched v1; consumers that only need task/run/attempt may read v1 legacy view **only** if documented as incomplete; must not infer `ExecutionId`. |

**Rationale:** NPSC-5F evidence plane goal is deterministic replay/correlation with the execution spine; optional `ExecutionId` would preserve ambiguous lineage and violate fail-closed enterprise rules.

### `platform_causal_evidence.v2` schema policy

| Rule | Decision |
| ---- | -------- |
| v1 readable? | **Yes** — legacy read supported where deserialization is safe |
| v1 write | **No** after cutover implementation — **write v2 only** |
| v1 lifecycle | **read-only / migration-only** → eventual **retired** for write paths |
| Dual-read | **Required** in implementation while historical v1 may exist |
| Dual-write | **No** — single canonical write version (v2) |
| Migration | **Yes** — explicit, provider-neutral migration layer (not in this task) |
| Contract versioning | **Existing frozen v1 semantics are not overwritten**; v2 is a new explicit version |

**Enterprise compatibility rule (formal):**

```text
read v1 + v2
write v2 only
```

### No silent fallback (forbidden)

```text
missing ExecutionId → mint new one          FORBIDDEN
missing ExecutionId → guess from AttemptId  FORBIDDEN
missing ExecutionId → ignore                  FORBIDDEN
```

### Legacy v1 causal evidence without `ExecutionId`

| Mode | Allowed? | Use |
| ---- | -------- | --- |
| **A.** Read as legacy incomplete evidence | **Yes** | Diagnostics, degraded reconstruction, audit with explicit incompleteness flag |
| **B.** Deterministic enrichment from canonical source | **Yes** | Only when a **unique** `ExecutionId` is provable from run/attempt lineage, journal, or identity persistence — never heuristics |
| **C.** Offline migration | **Yes** | Batch/idempotent jobs with audit trail |
| **D.** Reject | **Yes** | When canonical source cannot establish `ExecutionId` safely — **fail closed** |

Models that **guess** identity are **excluded**.

### Deterministic enrichment sources (if B is used)

Approved canonical sources only:

- Run/attempt lineage from execution identity authority records
- Durable journal / runtime events with bound `ExecutionId`
- `intergrax.bg_exec_identity.v2` (or v1 only when 4th field can be resolved uniquely from authority — otherwise fail closed)

### Background execution identity `v1` vs `v2`

| Partition | Semantics |
| --------- | --------- |
| `intergrax.bg_exec_identity.v1` | Legacy 3-field encoding; **read** supported in implementation; **no new writes** after cutover |
| `intergrax.bg_exec_identity.v2` | 4-field encoding including `execution_id`; **sole write target** after cutover |

**Partition/version migration decision (implementation task):**

| Mechanism | Decision |
| --------- | -------- |
| Dual-read | **Yes** — read v1 + v2 |
| Dual-write | **No** |
| Migrate-on-read | **Optional** — allowed if deterministic, idempotent, auditable |
| One-time migration | **Allowed** — offline or controlled job |
| Hard cutover | **Only** after compatibility PASS and operator sign-off — not default |

**Fail-closed:** If v1 record cannot safely yield `ExecutionId`, raise typed compatibility error — do not create synthetic identity.

### Typed migration error (implementation contract)

Stable boundary error (name fixed for design; implementation follow-on):

```text
LegacyBackgroundExecutionIdentityIncompatibleError
```

(or equivalent typed, provider-neutral error in `intergrax/contracts/` migration surface — **not** bare `RuntimeError`).

### Export boundary (`CausalEvidenceExportSource`)

| Topic | v2 policy |
| ----- | --------- |
| `target_execution_id` | **Required** on v2 export contract |
| v1 consumers | Continue to consume **versioned** v1 export envelopes until retired; no silent shape change |
| Exporter | May emit **both** schema versions during transition; **version field must be explicit** |
| Compatibility | **Versioned export contract** — no silent shape mutation |

---

## Identity Authority Invariants

Confirmed **unchanged**:

```text
Execution identity authority remains canonical
Observability does not mint ExecutionId (production required-audit path)
Evidence does not steer execution
Governance remains unchanged
ExecutionRuntime remains canonical owner
Retry/recovery ownership unchanged
```

Any future design that violates the above → **STOP** and escalate to full reopen review.

---

## Evidence/Control Invariants

- `PlatformCausalEvidence` remains a **record** of transport→execution relation.
- No scheduler, retry, policy, or recovery steering added via evidence or export.
- Dependency direction: **execution identity → persistence → evidence → export** (read paths may enrich from canonical stores only).

---

## Compatibility Policy

### Contract versioning rule

```text
existing frozen contract change → explicit new version
```

Do not semantically redefine `platform_causal_evidence.v1` or v1 export envelopes.

### Version lifecycle

| Version | State (target after implementation) |
| ------- | ----------------------------------- |
| `platform_causal_evidence.v1` | **deprecated** → **read-only** → **migration-only** → write **retired** |
| `platform_causal_evidence.v2` | **supported** (canonical write) |
| `intergrax.bg_exec_identity.v1` | **read-only** + migration |
| `intergrax.bg_exec_identity.v2` | **supported** (canonical write) |
| Export v1 envelope | **read-only** for consumers until retired |
| Export v2 envelope | **supported** |

---

## v1→v2 Migration Policy

**This task:** design authority only — **no** production migration code.

**Implementation task must:**

1. Introduce provider-neutral migration contracts (`contract → migration provider → vendor implementation`).
2. Avoid `getattr`/`setattr` architecture dispatch and hidden global registries; use composition.
3. Use typed, explicit contracts — no `Any`, no loose dicts for identity fields.
4. Ensure migrations are **deterministic, repeatable, idempotent, observable, auditable**.
5. Never overwrite unknown `ExecutionId`, forge lineage, or merge run/attempt heuristically.

### Migration test matrix (required in implementation + requalification)

| Case | Expectation |
| ---- | ----------- |
| v1 persisted → current reader | Legacy read or typed enrichment; no guess |
| v2 persisted → current reader | Full correlation |
| v1 + v2 coexist | Dual-read; write v2 only |
| invalid v1 | Fail closed / typed error |
| corrupted v2 | Fail closed |
| missing `execution_id` on v1 causal | Incomplete (A) or enrichment (B) or reject (D) — never mint |

---

## Persistence Compatibility Matrix

| Artifact | v1 Exists? | v2 Exists? | Read v1? | Write v1? | Write v2? | Migration Required? |
| -------- | ---------: | ---------: | -------: | --------: | --------: | ------------------: |
| Causal evidence (`PlatformCausalEvidence`) | Yes (historical) | Yes (post-`962`) | **Yes** (legacy) | **No** (post-cutover) | **Yes** | **Yes** |
| Background execution identity | Yes | Yes (post-`962`) | **Yes** | **No** (post-cutover) | **Yes** | **Yes** |
| Export boundary payload | v1 shape | v2 + `target_execution_id` | **Yes** | **No** | **Yes** | **Yes** (consumer/version) |

Current tree @ `962bf1ade`: v2 write path exists; **v1 read/migration not yet qualified** — sentinels correctly **RED**.

---

## Consumer Compatibility Matrix

| Consumer | Reads v1 | Reads v2 | Needs Change? |
| -------- | -------: | -------: | -------------: |
| Evidence reconstruction | Yes (incomplete) | Yes | **Yes** — handle version + optional incompleteness |
| Diagnostics | Yes | Yes | **Yes** — flag legacy incomplete |
| Exporter (`causal_evidence_export`) | Transition | Yes | **Yes** — explicit version field |
| Persistence adapters (causal + bg identity) | Yes | Yes | **Yes** — dual-read, write v2, migration provider |
| Background reentry | v1 records fail today | v2 | **Yes** — migration or fail-closed |
| Audit tooling | Yes | Yes | **Yes** — document incompleteness |

---

## Requalification Scope

After migration/compatibility implementation:

### NPSC-5F gate families

- **P0** reconciliation
- **R1** persistence (regression where shared stores touched)
- **R3** export
- **Final** evidence plane protected drift

### Identity

- Single authority gate
- **EE-A2** (as applicable to identity propagation)

### Persistence

- Background identity compatibility suite
- Causal evidence v1/v2 matrix tests

### Tracing / public contracts

- If export/public DTO surfaces change during implementation — impacted tracing/public contract gates per guard matrix

**Not in scope:** full platform revalidation at global baseline — only NPSC-5F R3 + Final scoped re-freeze.

---

## Protected Fingerprint Policy

**In this task:** **no** protected fingerprint or sentinel SHA updates.

Updates only after:

1. Implementation complete
2. Compatibility PASS
3. Regression PASS
4. Independent audit
5. Formal requalification / resignoff record

---

## Re-freeze Scope

Qualified baseline advance **only** for:

```text
NPSC-5F R3
NPSC-5F Final
```

**Not** for:

```text
a185403d0c7524c29bea2fe09212f9508e6bccd8 (global frozen code baseline)
```

---

## Architecture Risks

| Severity | Risk | Mitigation |
| -------- | ---- | ---------- |
| Critical | Migration invents `ExecutionId` | Fail-closed + typed errors; canonical-only enrichment |
| Critical | Observability becomes mint authority | Required-audit path review in implementation gates |
| Critical | Evidence affects control | P0 reconciliation + static boundaries |
| Major | No v1 strategy | Formal dual-read / write-v2-only policy (this record) |
| Major | Destructive migration | Idempotent, auditable providers; no overwrite of unknown IDs |
| Major | Unversioned contract mutation | v1 frozen; v2 explicit; versioned export |
| Minor | Docs vs fingerprint skew | Re-freeze + resignoff after implementation |

---

## Findings

### Critical

- None in **reopen record** — drift direction aligns with single authority **if** migration does not synthesize IDs.

### Major

- `962bf1ade` shipped v2 without qualified migration/read path — **unqualified BREAKING** until follow-on work.
- R3 export and Final sentinels **expected RED**.

### Minor

- Hygiene items from classification (e.g. unused imports) — address in implementation task, not reopen record.

---

## Final Reopen Decision

Approval criteria check:

- Minimal scope — **yes**
- Ownership unchanged — **yes**
- v2 semantics explicit — **yes**
- v1 compatibility explicit — **yes**
- Migration does not guess identity — **yes** (policy)
- evidence ≠ control — **yes**
- observability does not mint — **yes** (production path)
- provider-neutral persistence migration — **yes** (required in implementation)
- requalification scope defined — **yes**
- global frozen baseline unchanged — **yes**

```text
SCOPED ARCHITECTURE REOPEN APPROVED
```

**Production changes in this task:** **NONE**

**Next task:** `INTEGRAx-NPSC-5F-MIGRATION-COMPATIBILITY-IMPLEMENTATION` (or equivalent) — migration/compatibility design → implementation → requalification → protected-drift resignoff.

---

## Roadmap alignment

| Stage | Status |
| ----- | ------ |
| Platform Assurance Closure | CLOSED |
| Protected Drift Classification (Class C) | CLOSED |
| **Scoped NPSC-5F Architecture Reopen** | **CLOSED (this record)** |
| Migration / Compatibility Implementation | NEXT |
| NPSC-5F R3 + Final Requalification | NEXT |
| Current HEAD platform revalidation | FUTURE |

---

Wprowadzone zmiany wymagają audytu na podstawie aktualnego kodu znajdującego się na GitHub. Raport implementacji nie zastępuje niezależnej weryfikacji zmianów wykonanych w repozytorium.
