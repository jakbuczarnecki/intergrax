# NPSC-5D/R1 — Final Qualification, Cross-Layer Governance Admission & Freeze

**Status:** `FROZEN / PASS`

**Verdict:** **PASS** (final freeze certification)

**Date:** 2026-09-09

**Branch:** `development`

**Task:** NPSC-5D/R1 Final — Cross-Layer Governance Admission Qualification

**Final freeze task START SHA:** `cdad48bde50ecaa06670d6c8c87291ee7036b043`

---

## Purpose

NPSC-5D/R1 freezes the **stage-1 cross-layer governance admission contract** for multi-agent coordination. It is **not** per-contribution governance, physical delegation authorization, or HITL resume E2E.

R1 answers: **whether coordination may proceed** at the `CoordinationIntentExecutor` boundary, after authoritative host applicability reconciliation and before frozen NPSC-5A / NPSC-5B execution.

```text
CoordinationIntent
    ↓
validated runtime binding
    ↓
authoritative governed host applicability reconciliation
    ↓
MultiAgentCoordinationGovernancePort
    ↓
if collaborative REQUIRED:
    CollaborativeWorkAuthorityResolverPort
    ↓
    EffectiveAuthorityDecision
    ↓
coordination policy decision
    ↓
compose_policy_decisions
    ↓
final PolicyDecision
    ↓
ALLOW
    ↓
NPSC-5A or NPSC-5B
```

On any fail-closed outcome: **no downstream coordination execution**.

---

## Baseline / provenance

| Label | SHA |
| ----- | --- |
| `origin/development` baseline (task start) | `cdad48bde50ecaa06670d6c8c87291ee7036b043` |
| R1 foundation | `d63634db8b8318f1e2600fe0cc1416f4d3942d40` |
| R1-H1 collaborative authority reconciliation | `25260b156e8cf4ab06887355a9bd95aa622731ae` |
| R1-H2 authoritative applicability | `2fea88980e5959cc05028822f1c43ee527e627b3` |
| R1-H2-F1 missing governed context fail-closed | `2480db6e98e035af7525b03c0014d2ac3927d04e` |

**POST-F1 unrelated commits** (do not block R1 freeze):

```text
VPI vector-storage bootstrap integration tests only
```

No shared NPSC / Governance / Collaborative Work / Execution / Nexus / Decision seam changed after F1.

---

## Implementation chain (frozen)

| Artifact | Module | Role |
| -------- | ------ | ---- |
| `MultiAgentCoordinationGovernanceRequest` | `intergrax/contracts/multi_agent_coordination_governance.py` | Typed stage-1 admission facts |
| `MultiAgentCoordinationGovernancePort` | contracts | Public evaluator boundary |
| `MultiAgentCoordinationGovernanceBoundary` | `intergrax/runtime/governance/multi_agent_coordination_governance.py` | Fail-closed admission |
| `build_multi_agent_coordination_governance_request` | `intergrax/agent_distribution/coordination_governance_adapter.py` | Caller adapter |
| `materialize_coordination_intent_binding` | `intergrax/agent_distribution/coordination_binding_materialization.py` | Runtime binding projection |
| `CoordinationIntentExecutor` | `intergrax/agent_distribution/coordination_intent_executor.py` | Canonical evaluation location |
| `CollaborativeWorkAuthorityResolverPort` | `intergrax/autonomous_work/execution_authority_admission.py` | Shared consumer seam (AW-3B) |
| `compose_policy_decisions` | `intergrax/runtime/policy/` | Canonical composition only |

---

## Ownership table

| System | Owns |
| ------ | ---- |
| **Decision System** | WHAT work is required |
| **Agent Distribution / NPSC** | WHO can perform it (discovery, selection, lease, coordination) |
| **Collaborative Work** | WHO MAY ACT FOR WHOM (membership, delegation, effective authority) |
| **Governance** | WHETHER operation may proceed |
| **Execution** | Effective execution authority + lifecycle |
| **Nexus** | HOW / WHEN FAN_OUT executes |
| **HITL** | Human authority when Governance requires continuation |

**Never:**

- Decision evaluates permission
- NPSC resolves membership / delegation / effective authority
- NPSC mints `ExecutionId` or changes lifecycle
- Nexus submits topology outside frozen NPSC-5B
- Governance chooses physical specialist or scheduling order

---

## Security invariants (frozen)

| Invariant | Status |
| --------- | ------ |
| Semantic governance admission | PASS |
| Authoritative applicability from governed host Task | PASS |
| Missing governed context fail-closed | PASS |
| Collaborative authority composition via `compose_policy_decisions` | PASS |
| `RequestIdentity` ≠ `EffectiveAuthorityDecision` ≠ `PolicyDecision` | PASS |
| NPSC delegation ≠ Collaborative Work authority delegation | PASS |
| `PolicyAction.ALLOW` does not expand execution authority | PASS |
| Authority non-amplification (membership, delegation, scope) | PASS |
| Zero downstream effects on DENY / indeterminate | PASS |
| No second governance / authority / policy composer engine | PASS |
| No local HITL loop in NPSC | PASS |

---

## Applicability semantics (canonical)

Authoritative source: **governed Execution / host Task context** — not `CoordinationIntent`, Decision artifact, caller boolean, or binding omission.

| Host context | Applicability |
| ------------ | ------------- |
| governed Task + workspace | `REQUIRED` |
| governed Task, workspace absent | `NOT_APPLICABLE` |
| governed Task absent | `INDETERMINATE` / fail-closed |
| workspace present but malformed/empty | `INDETERMINATE` / fail-closed |

`CoordinationIntentBinding` is a **runtime projection**, not source of truth. Executor reconciliation with authoritative host context is mandatory.

---

## Authority composition semantics

For collaborative `REQUIRED`:

```text
CollaborativeWorkAuthorityResolverPort → EffectiveAuthorityDecision
EffectiveAuthorityDecision + coordination PolicyDecision → compose_policy_decisions
```

Precedence (canonical, reused):

| Authority | Policy | Result |
| --------- | ------ | ------ |
| DENY | ALLOW | DENY |
| ALLOW | DENY | DENY |
| ALLOW | REQUIRE_HUMAN | REQUIRE_HUMAN |
| DENY | REQUIRE_HUMAN | DENY |

`REQUIRE_HUMAN` blocks execution before approval; full pause/resume E2E is **deferred**.

---

## Delegation distinction

```text
DelegatedSubtaskService  → NPSC specialist contribution delegation
AuthorityDelegation      → Collaborative Work authority delegation
```

- `DelegatedSubtaskService` does **not** grant `AuthorityDelegation`
- `AuthorityDelegation` does **not** start child `Execution`

---

## Port ownership — `CollaborativeWorkAuthorityResolverPort`

Location: `intergrax/autonomous_work/execution_authority_admission.py`

**Valid because:** shared consumer seam from AW-3B admission composition — not Autonomous Work authority ownership. Collaborative Work owns resolver semantics; port is a stable cross-layer contract.

---

## Test matrix (final qualification)

| # | Scenario | Result |
| - | -------- | ------ |
| 1 | SINGLE non-collaborative ALLOW | PASS |
| 2 | SINGLE collaborative ALLOW | PASS |
| 3 | SINGLE collaborative authority DENY | PASS |
| 4 | FAN_OUT non-collaborative ALLOW | PASS |
| 5 | FAN_OUT collaborative ALLOW | PASS |
| 6 | FAN_OUT governance DENY | PASS |
| 7 | missing governed context DENY | PASS |
| 8 | workspace omission attack DENY | PASS |
| 9 | workspace substitution attack DENY | PASS |
| 10 | REQUIRE_HUMAN blocks downstream | PASS |
| 11 | Decision-backed path same boundary | PASS |
| 12 | deterministic producer same boundary | PASS |

**Cross-layer authority matrix:** valid membership, revoked membership, valid delegation, expired/revoked delegation, scope amplification — covered in focused suites + final qualification.

**Final qualification test:** `tests/unit/runtime/architecture/test_npsc5d_r1_final_qualification.py`

---

## Regression matrix

| Suite | Status |
| ----- | ------ |
| NPSC-5A coordination / delegation | PASS |
| NPSC-5B fan-out / fan-in | PASS |
| NPSC-5C decision projection / E2E | PASS |
| Collaborative Work authority / membership / delegation | PASS |
| Execution authority regression | PASS |
| Governance / policy composition | PASS |
| HITL REQUIRE_HUMAN / GovernanceResolution | PASS |

---

## Out of scope (R1)

| Item | Owner |
| ---- | ----- |
| Physical agent authorization | NPSC-5D/R2 |
| Per-contribution governance | NPSC-5D/R2 |
| Post-selection policy | NPSC-5D/R2 |
| Lease-specific governance | NPSC-5D/R2 |
| Canonical HITL governed continuation E2E | NPSC-5D/R3 |
| Scheduling / parallelism / retry order | Nexus |
| Provider / specialist selection | Agent Distribution |
| Durable governance evidence / replay | Later |

---

## Known deferred items

```text
NPSC-5D/R2: physical delegation / per-contribution governance
NPSC-5D/R3: canonical HITL governed continuation E2E
NPSC-5D Final: complete governance qualification/freeze
```

---

## Formal verdict

```text
NPSC-5D/R1 = FROZEN / PASS
NPSC-5D     = ACTIVE
```

Production code unchanged in this qualification task. Entire R1 chain proven together with explicit ownership boundaries, authoritative applicability, fail-closed missing context, and zero downstream effects on deny.

---

## Freeze statement

> **NPSC-5D/R1 is exclusively stage-1 governance admission: authoritative host context determines applicability, Collaborative Work determines collaborative authority, Governance determines WHETHER, Execution preserves authority/lifecycle, and only after ALLOW may frozen NPSC-5A/NPSC-5B execute coordination.**

R1 frozen ≠ NPSC-5D frozen. R2 and R3 remain active under NPSC-5D.

---

## Next task

**NPSC-5D/R2 — Physical Delegation / Contribution-Level Governance**
