# DECISION_SYSTEM — platform boundaries

**Parent hub:** [`DECISION_SYSTEM.md`](../DECISION_SYSTEM.md)

## 1. Nexus

Sole execution owner for Decision Lifecycle stages, budgets, checkpoints, technical retry, and persistence.

## 2. Policy / Governance

Cross-cutting execution authorization — evaluates consequential actions; does not determine decision correctness.

## 3. HITL

Canonical human approver / adjudicator / escalation records — invoked by lifecycle, not implemented inside Decision System.

## 4. Reliability

Technical retry on provider/tool failure — distinct from semantic revision and deliberation rounds.

## 5. Observability

Records decision audit evidence — strategy, versions, verification, resolution, authorization relation. No private CoT.

## 6. Diagnostics

May inform investigation flows — does not own lifecycle or rubric content.

## 7. Evidence Claims / Eval

Evidence claims support evidence-backed decisions. Online/shadow/offline eval remain **outside** runtime verification ownership.

## 8. Tools / LLM adapters

Invoked by strategies and verification stages under governed tool/LLM boundaries.
