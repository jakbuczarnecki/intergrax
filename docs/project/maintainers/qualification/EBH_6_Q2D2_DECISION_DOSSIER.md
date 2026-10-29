# EBH-6 — Q2-D2 Decision Dossier (E03)

| Field | Value |
| --- | --- |
| **Operator decision** | **OD-EBH6-Q2D2** — **OPEN** (agent must not close) |
| **Finding** | **EBH6-DEBT-Q2-D2** — register **RATIFIED** · AJ **`semantic_core`×F = FINDING** |
| **Invariant** | **FRZ-TYP-03** |
| **Production surface** | `intergrax/runtime/nexus/nexus_loop.py` — `AuthoritativeDecisionExposure` typing seam |
| **SSOT** | [`_findings_register.py`](../../../tests/qualification/ebh_6/_findings_register.py) · bundle `q2d2_finding` |
| **Baseline SHA** | `202239f726dd5e3fe23bf1a68cafa61c1eb9fca7` |

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

## STOP

Only the operator may set **OD-EBH6-Q2D2** to APPROVED. Agent must not resolve this decision.
