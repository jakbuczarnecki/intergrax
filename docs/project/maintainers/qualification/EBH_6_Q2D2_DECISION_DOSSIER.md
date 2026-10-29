# EBH-6 — Q2-D2 Decision Dossier (E03)

| Field | Value |
| --- | --- |
| **Operator decision** | **OD-EBH6-Q2D2** — **OPEN** (agent must not close) |
| **Finding** | **EBH6-DEBT-Q2-D2** — register **RATIFIED** · AJ **`semantic_core`×F = FINDING** |
| **Invariant** | **FRZ-TYP-03** |
| **Production surface** | `intergrax/runtime/nexus/nexus_loop.py` — `AuthoritativeDecisionExposure` typing seam |
| **SSOT** | [`_findings_register.py`](../../../tests/qualification/ebh_6/_findings_register.py) · bundle `q2d2_finding` |
| **Baseline SHA** | `05e3363c5382532fd4d5ed4990b610f342ef5834` (qual HEAD) · substance `c852e770…` |

## Evidence on record

| Artifact | Role |
| --- | --- |
| Register row EBH6-DEBT-Q2-D2 | RATIFIED · REQUIRES ADR · production_effective **YES** |
| Class F gate (`semantic_core`×F) | **FINDING** outcome — not PASS_CANDIDATE |
| T-CLS-F probe (read-only nexus seam) | Qual negative / honesty — does not auto-promote to PASS |
| Family **TYP** row | PENDING independent — cites Q2-D2 disposition |

## Path A — Scoped ratification acceptance (operator)

| Step | Action | Owner |
| --- | --- | --- |
| A1 | Document prior ratification scope vs current EBH-6 obligation | Operator + auditor |
| A2 | Confirm qual gates prove **no new** untyped public seam beyond registered debt | Independent auditor |
| A3 | Record **OD-EBH6-Q2D2 APPROVED — PATH A** | Operator |
| A4 | Auditor accepts FINDING track for P03 as **scoped PASS** on TYP (not silent AJ promotion) | Independent auditor |

**Does not:** auto-issue Protocol 03 PASS or flip implementer FINDING without auditor row in P03 report.

## Path B — Production remediation

| Step | Action | Owner |
| --- | --- | --- |
| B1 | **OD-EBH6-01** — wave open phrase recorded | Operator |
| B2 | Narrow typing on nexus public seam per FRZ-TYP-03 | Implementer (prod) |
| B3 | Regression: class F gate + TYP family gates @ new SHA | Agent/implementer |
| B4 | Close FINDING → PASS_CANDIDATE only with auditor acceptance | Independent auditor |

## Implementer recommendation (@ baseline evidence)

**Recommend Path A** for EBH-6 closure **if** independent pre-cert (E01) confirms: (1) ratification scope matches register, (2) adversarial negatives bound production impact to registered debt, (3) no additional PROHIBITED typing violations on semantic_core closed world.

**Recommend Path B** if auditor rejects ratification-as-sufficient for **production_effective YES** + **REQUIRES ADR** classification without signed ADR acceptance artifact.

## R3 — contract verification (@ HEAD)

| Check | Finding |
| --- | --- |
| Surface | `resolve_authoritative_decision_exposure_for_task` → `AuthoritativeDecisionExposure[object]` in [`nexus_decision_exposure.py`](../../../intergrax/runtime/nexus/orchestration/nexus_decision_exposure.py) |
| `object` erasure | **Yes** — parametric exposure erases candidate typing; selection still typed via `DecisionExposureSelectionHostBinding` + fail-closed errors |
| Authority | Terminal states require `effective_attempt_id`; missing → `NexusDecisionExposureError` (not silent fallback) |
| Publication | `ExposureUnevaluated` / selection failure paths explicit |
| FRZ-TYP-03 | Strict typing **not met** — debt track only via ratification |
| ADR | Operator must confirm ADR scope covers **current** exposure resolver @ HEAD |

**Recommendation:** **PATH A — ACCEPT SCOPED DEBT** for EBH-6 **if** operator OD-EBH6-Q2D2 + auditor accept FINDING track; else **PATH B** + **OD-EBH6-01**.

## C3-01 — Path A audit (implementer analysis @ HEAD)

| # | Question | Finding |
| --- | --- | --- |
| 1 | Historical ratification scope | Register notes **EBH-6 wave** ratification; ADR-class **REQUIRES ADR** — scope is **nexus decision exposure typing**, not whole semantic_core layer |
| 2 | Current production contract | Primary seam: `AuthoritativeDecisionExposure[object]` in [`nexus_decision_exposure.py`](../../../intergrax/runtime/nexus/orchestration/nexus_decision_exposure.py) — bounded to decision exposure publication path |
| 3 | Authority / fail-closed | Exposure path uses explicit `NexusDecisionExposureError` failure codes — debt is **typing**, not bypass of governance selection |
| 4 | Negative tests | Class F FINDING gate + register `test_ebh6_tracked_debt_findings_ratified` — no silent PASS |
| 5 | FRZ-TYP-03 alignment | Debt **violates strict FRZ-TYP-03** — accepted only via **scoped ratification**, not invariant rewrite |
| 6 | ADR coverage | Operator must confirm ADR artifact covers **current** `nexus_decision_exposure` surface @ HEAD |
| 7 | PROHIBITED | **0** unresolved PROHIBITED @ HEAD — Q2-D2 is **tracked debt**, not PROHIBITED-* |

## C3-02 — Path B readiness (not executed)

| Item | Proposal |
| --- | --- |
| Symbols | Narrow `AuthoritativeDecisionExposure[object]` → typed protocol bound to sanctioned decision exposure contracts |
| Callers | `task_finisher.py`, `graph_runner.py`, `nexus_loop.py` selection host binding |
| Regression | `semantic_core`×F class gate · family **TYP** gates · EE/nexus integration subset |
| Authorization | **OD-EBH6-01** mandatory before prod edit |

## C3-03 — Recommendation

**PATH A — ACCEPT SCOPED DEBT** (operator + auditor), contingent on E01 confirming no expanded untyped surface beyond register.

Alternative: **PATH B — PRODUCTION REMEDIATION REQUIRED** if ADR scope or auditor rejects ratification for production_effective YES seam.

## C3-04 — Operator gate (proposed text — NOT APPROVED)

```text
OPERATOR DECISION — OD-EBH6-Q2D2 — PENDING
Option A: APPROVED — EBH6-DEBT-Q2-D2 scoped ratification satisfies EBH-6 Protocol 03 FINDING track.
Option B: ORDER REMEDIATION — Path B + OD-EBH6-01 before re-audit.
```

## STOP

Only the operator may set **OD-EBH6-Q2D2** to APPROVED. Agent must not resolve this decision.

