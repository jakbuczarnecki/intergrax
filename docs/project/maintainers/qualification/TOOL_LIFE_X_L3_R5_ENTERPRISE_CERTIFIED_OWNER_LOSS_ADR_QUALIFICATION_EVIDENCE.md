# TOOL-LIFE-X L3-R5 — Enterprise Certified Owner-Loss & STALLED Policy ADR Evidence

| Field | Value |
| --- | --- |
| **Task** | `TOOL-LIFE-X-L3-R5` (remediation batch R5-F01–F04) |
| **Parent** | `TOOL-LIFE-X` (§3.0.5) — **BLOCKED** / **not CLOSED** |
| **START_HEAD** | `dd63629d75f28ce25e45c07c90e52a59bce814fe` |
| **ADR** | [`ADR_ENTERPRISE_TOOL_LIFE_X_CERTIFIED_OWNER_LOSS_AND_STALLED_POLICY.md`](../architecture/ADR_ENTERPRISE_TOOL_LIFE_X_CERTIFIED_OWNER_LOSS_AND_STALLED_POLICY.md) |
| **Production delta** | **0** |
| **pytest @ this task** | **0** (architecture/documentation; static validation only) |
| **Classification** | **B — ARCHITECTURE DECISION REQUIRED** |
| **Disposition** | **`TOOL-LIFE-X L3-R5 — READY FOR INDEPENDENT AUDIT`** |

## Scope

Independent audit remediation R5-F01–F04 in one ADR package: certificate trust boundary, store-native atomicity, tenant-safe persistence contract, closed-world authority inventory, §11.2 negative-case matrix, TL1–TL15 closure matrix, ECAP/FRZ scoped notes, tenant isolation audit (local FAIL until store protocol). **No** L3-R6 implementation.

## R5 disposition summary

| Finding | Disposition |
| --- | --- |
| **R5-F01** authenticity | `evidence_digest` = integrity only; authenticity = staged evidence in store + issuer allowlist + one-time consumption in `apply_certified_owner_loss` CAS — **OPERATOR DECISION REQUIRED** (§20.1 B1) |
| **R5-F02** atomicity | Reject `verify()` then `compare_and_set()`; require single CAS linearization — **ADR corrected**; impl **BLOCKED** until L3-R5-B |
| **R5-F03** tenant | Durable store not tenant-keyed @ HEAD — mandatory `tenant_id` + composite API — **OPERATOR DECISION REQUIRED** |
| **R5-F04** issuer | **No** sanctioned issuer @ HEAD; Path B deny preserved — **B** blocks L3-R6 |

## Prior waves reused

| Wave | Evidence |
| --- | --- |
| P0 | [`TOOL_LIFE_X_P0_CANONICAL_LIFECYCLE_CLOSED_WORLD_PRE_AUDIT.md`](TOOL_LIFE_X_P0_CANONICAL_LIFECYCLE_CLOSED_WORLD_PRE_AUDIT.md) |
| Batch C | [`TOOL_LIFE_X_BATCH_C_STALLED_ABANDONED_RECOVERY_EVIDENCE.md`](TOOL_LIFE_X_BATCH_C_STALLED_ABANDONED_RECOVERY_EVIDENCE.md) (Path B C-F04) |
| Batch E | [`TOOL_LIFE_X_BATCH_E_ADVERSARIAL_REAL_INFRASTRUCTURE_EVIDENCE.md`](TOOL_LIFE_X_BATCH_E_ADVERSARIAL_REAL_INFRASTRUCTURE_EVIDENCE.md) |
| L3-R1 / L3-R2 | consolidated + MCP host qual |

## Static validation (@ publication)

| Check | Result |
| --- | --- |
| `recover_running_operation_after_certified_owner_loss` fail-closed on RUNNING | **PASS** — `intergrax/runtime/external_operations/external_operation_ownership.py` |
| `ExternalOperationState` / store — no `tenant_id` @ HEAD (documented gap) | **PASS** (honest inventory) |
| `ExternalOperationStateStore` — `operation_id`-only key | **PASS** |
| `mint_stable_operation_id` — no tenant in digest input | **PASS** |
| No `CertifiedOwnerLossEvidenceIssuerPort` implementation | **PASS** |
| STALLED DEADLINE_ONLY helper | **PASS** — `tool_operation_recovery.py` |
| ADR §20 matches code inventory | **PASS** |

## ECAP / FRZ (scoped)

| ID | Note |
| --- | --- |
| ECAP P1–P4 | Docs-only task — no self-PASS |
| FRZ-BND-03 | Single lifecycle owner preserved |
| FRZ-REL-05 | OPEN — input to L3-R7 after implementation |
| FRZ-REC-07 | UNKNOWN modeling aligned; no global PASS |
| FRZ-TEN (local) | Shared ext-op store tenant keying **OPEN** until L3-R6 protocol |

## Tenant Isolation Audit (L3-R5 local)

**Result:** **FAIL** for certified owner-loss on shared durability until tenant-scoped store contract ships. Admission-layer tenant checks do not protect bare `operation_id` load/CAS. **Does not** reopen global TENANT-X closure.

## Post-step roadmap audit

- **TOOL-LIFE-X** parent remains **BLOCKED**; **not CLOSED**.
- **Next:** operator ratification **L3-R5-B** (ADR §20.4 Option B1); then **L3-R6** implementation.
- **DIST-SCALE-X:** not started.
- TL10–TL13 physical qual blockers **preserved**.

## Test-selection record

1. Production behavior changed? **No**
2. pytest required? **No**
3. Reuse Batch C deny @ historical SHAs? **Yes**

## PUBLICATION_SHA

Git commit on `development` containing this remediation (roadmap §5 ledger **TOOL-LIFE-X**).

## Implementation eligibility

**L3-R6:** **prohibited** until operator architecture decision record ratifies ADR §20.4 Option B1.
