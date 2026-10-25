# TOOL-LIFE-X L3-R5-B — Architecture ratification (B1) & Gate A prequalification

| Field | Value |
| --- | --- |
| **Task** | `TOOL-LIFE-X-L3-R5-B` |
| **Authority** | `OPERATOR APPROVED — L3-R5-B OPTION B1 — ARCHITECTURE RATIFICATION ONLY` |
| **START_HEAD** | `141aaa3f8260929cd861e761f11739a753d9ba5d` |
| **Branch** | `development` |
| **Repository** | `jakbuczarnecki/intergrax` |
| **Production delta** | **0** |
| **pytest @ this task** | **0** (L1 static only; L2 reuse per §10) |
| **ADR** | [`ADR_ENTERPRISE_TOOL_LIFE_X_CERTIFIED_OWNER_LOSS_AND_STALLED_POLICY.md`](../architecture/ADR_ENTERPRISE_TOOL_LIFE_X_CERTIFIED_OWNER_LOSS_AND_STALLED_POLICY.md) §21 |
| **Disposition** | **`L3-R5-B RATIFICATION RECORD + GATE A PREQUALIFICATION — READY FOR INDEPENDENT AUDIT`** |

## Operator ratification record (canonical)

| Dimension | Status |
| --- | --- |
| **B1 architecture** | **RATIFIED** — tenant-scoped store, single semantic owner, store-native stage/apply CAS, composition-bound issuer, durable revocation/fencing, certified owner-loss, fail-closed defaults |
| **Implementation evidence R6–R10** | **Present @ HEAD** — **not** blanket-accepted by ratification; per-wave independent audit status preserved |
| **Gate A** | **NOT PROVEN** — recommendation **FAIL/BLOCKED** |
| **L3 qualification** | **INCOMPLETE** |
| **Positive production recovery** | **NOT AUTHORIZED** |
| **TL7** | **DEADLINE_ONLY** |
| **TL8** | **BLOCKED** (positive path) |
| **TOOL-LIFE-X parent** | **BLOCKED** |
| **DIST-SCALE-X** | **Not entered** |
| **new global FRZ PASS** | **0** |

**Historical reconciliation:** L3-R5 ADR §12/§20 stated **L3-R6 prohibited until L3-R5-B**. R6–R10 landed on `development` **before** this ratification publication. That sequence is **documented, not rewritten**: architecture direction is now ratified; **activation, host reachability, and physical effect-sink proofs remain gated**.

## R6–R10 reconciliation (@ START_HEAD)

| Wave | Evidence | Implementation @ HEAD | Ratification acceptance |
| --- | --- | --- | --- |
| **R6** | [`TOOL_LIFE_X_L3_R6_…`](TOOL_LIFE_X_L3_R6_TENANT_SAFE_EXTERNAL_OPERATION_FOUNDATION_EVIDENCE.md) | `tenant_id` on store API; certification bypass guards | Architecture-aligned; **VALIDATION REQUIRED** |
| **R7** | [`TOOL_LIFE_X_L3_R7_…`](TOOL_LIFE_X_L3_R7_CERTIFIED_OWNER_LOSS_ISSUER_EVIDENCE.md) | Issuer + stage/apply; adversarial pytest | **VALIDATION REQUIRED** |
| **R8** | [`TOOL_LIFE_X_L3_R8_…`](TOOL_LIFE_X_L3_R8_DURABLE_REVOCATION_SHARED_STORE_EVIDENCE.md) | SQLite colocated revocation + cross-process tests | **Scoped** single-host; multi-host **BLOCKED** |
| **R9** | [`TOOL_LIFE_X_L3_R9_…`](TOOL_LIFE_X_L3_R9_AUTHENTICATED_REVOCATION_PRODUCTION_COMPOSITION_EVIDENCE.md) | Gate B fail-closed; supervision MAC | **Gate A gap** documented in R9 |
| **R10** | [`TOOL_LIFE_X_L3_R10_…`](TOOL_LIFE_X_L3_R10_COMMIT_AUTHORITY_LOSS_DECISION_HOST_INTEGRATION_EVIDENCE.md) | Decision in revocation TX; host factory | **Factory-only** wiring; positive recovery disabled |

**L2 execution SHA (reuse, not rerun @ R5-B):** R10 evidence cites **86 passed** external_operations unit tests @ `ec2d74158…` lineage; applicability to `141aaa3f…` requires independent exact-SHA audit.

## Authority graph (@ HEAD — closed-world)

```text
Semantic owner (ext-op physical state, terminals, fencing):
  ExternalOperationStateStore + external_operation_ownership helpers
  Linearization: stage_certified_owner_loss_evidence / apply_certified_owner_loss / compare_and_set (non-cert fields)

Composition owner (issuer allowlist, revocation ports):
  certified_owner_loss_composition.CertifiedOwnerLossComposition (module singleton; empty allowlist default)

Worker execution certified loss staging:
  worker_execution_certified_owner_loss → store.stage_* + verify_worker_execution_owner_loss_staging

Revocation authority (durable fence for worker epoch):
  WorkerExecutionOwnershipRevocationPort (SQLite colocated @ DURABLE_SINGLE_HOST compose)

Revocation admission (production):
  FailClosedWorkerExecutionCommitAuthorityLossPolicy (R9/R10)

Supervision (MAC-bound authorization):
  WorkerExecutionOwnershipRevocationSupervisionPort + trust material (R9)

Production ToolRuntime default:
  runtime_tool_invoker_composition — optional in-memory store; **no** certified-owner-loss issuer binding

Sanctioned durable single-host factory (not production-bootstrapped @ HEAD):
  wire_tool_life_durable_single_host_execution_recovery → compose_tool_life_durable_single_host_certified_owner_loss
  Call sites: **tests only** (no Tier-3 application bootstrap reference)
```

**Second authority / bypass:** Ordinary `compare_and_set` cannot mutate certification or worker-execution binding fields (`CertifiedOwnerLossAuthorityBypassError`). Revoked worker epochs block non-cert CAS (`CertifiedOwnerLossEvidenceRejectedError`). **No** alternate dispatcher for RUNNING→UNKNOWN except `apply_certified_owner_loss` with staged evidence + allowlisted issuer.

**ADR baseline delta:** @ L3-R5 START_HEAD, §20.4 claimed **no sanctioned issuer**. @ `141aaa3f…`, issuer **exists** when composition wired (qualification/factory paths); **production default remains fail-closed** (empty `allowed_issuer_ids`).

## Gate A central question

> What existing canonical Execution/Recovery operation proves the previous worker **irreversibly lost** authority to commit or produce externally meaningful effects?

**Answer @ HEAD:** **No operation proven for Gate A closure.** Durable revocation + SQLite decision minting prove **internal commit-authority loss admission** on the **DURABLE_SINGLE_HOST** composition path when explicitly wired; **not** universal external effect-sink enforcement; **not** wired into canonical production host bootstrap.

### G-A01 — Internal durable state

| Finding | Classification |
| --- | --- |
| Revoked epoch blocks cert-adjacent CAS on store | **ALREADY PROVEN** (scoped) — R6/R8 unit tests |
| Every persistent write path tenant+epoch checked | **VALIDATION REQUIRED** — invoker paths outside SQLite qual store |
| Multi-host shared store | **BLOCKED** — topology guard fail-closed |

### G-A02 — External side effects

| Finding | Classification |
| --- | --- |
| Local SQLite revocation prevents stale **durable** commit | **ALREADY PROVEN** (scoped) |
| Provider API / network effect sink fenced | **MISSING** — no proof stale worker cannot call HTTP/MCP/DB |
| Path B reconciliation | **ALREADY PROVEN** (Batch B/L3-R1 scoped) — UNKNOWN/oracle; not owner-loss |

### G-A03 — In-flight operation

| Finding | Classification |
| --- | --- |
| Submitted-before-revoke external call | **OPERATOR DECISION REQUIRED** / **VALIDATION REQUIRED** — reconcile UNKNOWN; no auto SUCCESS |
| Lost ack after external commit | **ALREADY PROVEN** (scoped L3-R1) — remain UNKNOWN |

### G-A04 — Worker lifecycle vs authority

| Finding | Classification |
| --- | --- |
| STOPPED/RECOVERING/missing worker/deadline/heartbeat/lease → certified loss | **ALREADY PROVEN** deny — R9/R10 fail-closed policies |
| Lifecycle/repo injection admits loss | **CORRECTED** — R10-F06 removed |

### G-A05 — Trust

| Finding | Classification |
| --- | --- |
| Issuer allowlist + digest + consumption | **ALREADY PROVEN** (scoped R7) |
| Revocation MAC + supervision | **ALREADY PROVEN** (scoped R9) — **production composition default off** |
| Cross-tenant replay | **ALREADY PROVEN** reject (R6/R7 tests) |

### G-A06 — Durable races

| Finding | Classification |
| --- | --- |
| Terminal-first / revoke-first / stage-apply | **ALREADY PROVEN** (scoped R6/R8/R10) |
| Restart / WAL / partial persistence | **VALIDATION REQUIRED** — R8 cross-process subset; not full failure injection matrix |

### G-A07 — Host integration

| Finding | Classification |
| --- | --- |
| `wire_tool_life_*` in production bootstrap | **MISSING** — factory + tests only (R10-F04 OPEN) |
| Positive recovery reachable in production | **BLOCKED** by policy + empty issuer binding |

### G-A08 — Physical qualification

| Finding | Classification |
| --- | --- |
| SQLite single-host cross-process | **ALREADY PROVEN** (scoped R8) |
| Real provider / partition / multi-host | **MISSING** — L3-PHYSICAL partial; TL10–TL13 **BLOCKED** |

### G-A09 — Safe recovery semantics

| Action | Defensible @ evidence |
| --- | --- |
| Reconciliation without new side effects | **YES** (scoped Batch B) |
| Remain UNKNOWN | **YES** |
| Operator intervention | **YES** |
| Provider-status reconciliation | **VALIDATION REQUIRED** per provider |
| Positive resumption | **NOT AUTHORIZED** — not proven |

### G-A10 — Path B

Architecture-scoped: single owner, fail-closed defaults, no shadow authority — **aligned with B1 ratification**. Live provider / multi-instance obligations — **independent qualification**; Path B **does not** waive effect-sink or Gate A physical proofs.

**Gate A recommendation:** **FAIL/BLOCKED** — missing canonical production execution proof + external effect authority.

## Adversarial matrix (mandatory)

| Case | Canonical owner | Evidence | Expected (fail-closed) | Test @ HEAD | FRZ | Status |
| --- | --- | --- | --- | --- | --- | --- |
| Old worker alive but unreachable | Revocation + supervision | R8/R9 | No auto certified loss | Partial R8 | FRZ-REL-05 | VALIDATION REQUIRED |
| Old worker paused/resumed | Store owner_token CAS | R6/R7 | Fence stale owner | R7 tests | FRZ-REL-05 | ALREADY PROVEN (scoped) |
| Network partition | Store unavailable | ADR §11 | No guess; fail-closed | Limited | FRZ-REL-05 | VALIDATION REQUIRED |
| Crash before external effect | Invoker/idempotency | Batch B | Pre-effect paths only | Batch B | FRZ-REC-* | ALREADY PROVEN (scoped) |
| Crash after effect, before ack | Reconciliation | L3-R1 | UNKNOWN | L3-R1 | FRZ-REC-* | ALREADY PROVEN (scoped) |
| Revoke vs terminal commit | Store CAS | R8/R10 | Terminal precedence | R8/R10 | FRZ-REL-05 | ALREADY PROVEN (scoped) |
| Revoke vs cancellation | W4-C store | Batch C | Intent precedence | Batch C | FRZ-REL-04 | ALREADY PROVEN (scoped) |
| Two recovery candidates | apply_certified_owner_loss CAS | R7 | One CAS winner | R7 | FRZ-REL-05 | ALREADY PROVEN (scoped) |
| Revoked worker late durable CAS | compare_and_set guard | R6/R8 | Reject | R8 | FRZ-REL-05 | ALREADY PROVEN (scoped) |
| Revoked worker late external API | **No platform fence** | — | **Not proven blocked** | **Missing** | FRZ-REL-05/06 | **BLOCKED** |
| Cross-tenant evidence replay | stage/apply | R6 | Reject | R6 | FRZ-TEN-* | ALREADY PROVEN (scoped) |
| Stale epoch / operation id | stage/apply | R7 | Reject | R7 | FRZ-REL-05 | ALREADY PROVEN (scoped) |
| Forged issuer / MAC | composition + trust | R7/R9 | Reject | R9 | FRZ-SEC-* | ALREADY PROVEN (scoped) |
| Duplicate stage/apply | consumption set | R6 | Reject/idempotent | R6 | FRZ-REL-05 | ALREADY PROVEN (scoped) |
| Store restart / WAL | SQLite R8 | R8 | Consistent revocation | R8 cross-process | FRZ-STA-* | VALIDATION REQUIRED |
| Crash revoke→stage / stage→apply | Single TX / CAS | R10/R8 | Atomic boundaries | Partial | FRZ-REL-05 | VALIDATION REQUIRED |
| Partial durability loss | — | — | Fail-closed | **Missing** | FRZ-REL-05 | BLOCKED |
| Shared-store topology mismatch | topology guard | R8 | Fail-closed | R8 | FRZ-REL-05 | ALREADY PROVEN (scoped) |
| Host bootstrap w/o supervision | production default | R9 | Deny | R9 | FRZ-OWN-03 | ALREADY PROVEN (scoped) |
| Qual-only authority in production | empty issuer binding | R5-B audit | Deny | Batch C + default | FRZ-OWN-04 | ALREADY PROVEN (scoped) |

## Closure matrix (summary)

| Invariant | Owner | Contract | Proof | Evidence SHA | Missing action | FRZ | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Single lifecycle semantic owner | Ext-op store | W4-C + store protocol | Batch C + R6 | R6 publication | Independent HEAD audit | FRZ-BND-03 | VALIDATION REQUIRED |
| Exactly-one composition issuer | `certified_owner_loss_composition` | `CompositionCertifiedOwnerLossIssuerBinding` | Empty default | @ HEAD | Production wire when authorized | FRZ-OWN-03 | ALREADY PROVEN (scoped) |
| Store-native certification CAS | `ExternalOperationStateStore` | stage/apply | R6/R7 pytest | R7 lineage | exact-SHA audit | FRZ-REL-05 | VALIDATION REQUIRED |
| Durable revocation fence | SQLite revocation port | R8 contracts | R8 L2 | R8 lineage | Multi-host | FRZ-REL-05 | VALIDATION REQUIRED |
| Production reachability | Tier-3 bootstrap | `wire_tool_life_*` | Grep: tests only | R5-B | Host integration task | FRZ-EXE-* | **BLOCKED** |
| External effect authority | RuntimeToolInvoker | ToolContract | L3-R1 partial | L3-R1 | Provider physical qual | FRZ-REL-06 | **BLOCKED** |
| Physical worker termination | Platform | L3-PHYSICAL | Docker PG scoped | L3-PHYSICAL | MCP/TL11 | FRZ-REG-* | **BLOCKED** |
| Gate A closure | Execution/recovery | ADR §21 | This doc | R5-B commit | Canonical op + effect proof | FRZ-REC-* | **BLOCKED** |
| B1 architecture | Operator | ADR §21 | Ratification record | R5-B commit | Independent audit | ECAP P1–P4 | VALIDATION REQUIRED |

## ECAP / FRZ (scoped — applicable IDs @ task)

| ID | Disposition |
| --- | --- |
| **FRZ-BND-03** | No second lifecycle owner — **preserved**; tier boundaries unchanged |
| **FRZ-OWN-01..05** | Single semantic + composition owners on audited paths; shadow selection **denied** @ RAG unrelated; ext-op **VALIDATION REQUIRED** for production wire |
| **FRZ-CTR-01..06** | Contracts for cert loss / revocation — **implemented**; global rows remain **OPEN** per checklist |
| **FRZ-TYP-01..02** | Typed evidence/revocation models — **scoped green** in R6–R10 modules |
| **FRZ-EXE-*** | Historical PASS **not** ext-op host integration proof |
| **FRZ-GOV-*** | No new governance authority from R5-B |
| **FRZ-STA-*** | SQLite durability — partial R8 only |
| **FRZ-REC-*** | UNKNOWN modeling; positive recovery **not authorized** |
| **FRZ-REL-01..07** | TL7 DEADLINE_ONLY; TL8 positive **BLOCKED**; REL-05 **OPEN** for Gate A |
| **FRZ-SEC-*** | MAC/supervision scoped R9; production default fail-closed |
| **FRZ-TEN-*** | Tenant-keyed store @ R6 — **scoped**; checklist defaults **OPEN** |
| **FRZ-REG-*** | TL10–TL13 physical — **BLOCKED** (operator C10) |

**new global FRZ PASS = 0**

## L1 validation (@ this task)

| Check | Result |
| --- | --- |
| `git diff --check` | Run @ publication commit |
| Production code delta | **0** |
| ADR §21 + this record | Present |
| `wire_tool_life_*` production call sites | **0** (tests only) |

## Next permissible work (operator-gated)

1. **Independent exact-SHA audit** of R5-B publication + R6–R10 applicability @ `141aaa3f…`.
2. **Gate A implementation/qualification** (separate authorization): canonical execution/recovery operation + effect-sink proofs — **not** included in B1 ratification.
3. **Host integration** (if authorized): call `wire_tool_life_durable_single_host_execution_recovery` from sanctioned production bootstrap with explicit topology — still **no** positive recovery without separate approval.
4. **TL11 / L3-PHYSICAL follow-on** per roadmap — does **not** start DIST-SCALE-X.

## PUBLICATION_SHA

Git commit on `development` containing this record (operator ratification publication).
