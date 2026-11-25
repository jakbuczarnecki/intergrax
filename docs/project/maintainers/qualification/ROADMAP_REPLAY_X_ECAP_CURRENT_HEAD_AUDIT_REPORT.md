# ROADMAP-REPLAY-X — ECAP current-HEAD audit report (§14)

| Field | Value |
| --- | --- |
| **AUDITED_HEAD** | `343b32d62a5c9f225afc89d6b7ad5815279553d5` |
| **Parent status** | **CLOSED — INDEPENDENTLY ACCEPTED (SCOPED)** @ **OD-REPLAYX-08** (2026-10-30) |

## C1 — Closed-world inventory

[`ROADMAP_REPLAY_X_CLOSED_WORLD_INVENTORY.md`](ROADMAP_REPLAY_X_CLOSED_WORLD_INVENTORY.md) — Tier 0–3; unclassified **0** in replay scope.

## C2 — Ownership / duplicate authority

**Scoped production delta exists** (PRE-ENT L3 @ `01b3e196` + L2 CTRL-X + **Wave 06 EQO** 418 paths @ W06 head `da5ca5b53` — dispositions in `_pre_ent_l3_prod_delta.py`, `_w06_eqo_prod_delta.py`, `_prod_delta_disposition.py`; map [`PRE_ENT_REPLAY_L3_PROD_DELTA.md`](pre_ent_enterprise_certification/PRE_ENT_REPLAY_L3_PROD_DELTA.md) + [`W06_REPLAY_PROD_DELTA.md`](pre_ent_enterprise_certification/W06_REPLAY_PROD_DELTA.md); CTRL-X regression @ `22fbc0e` preserved). **Unscoped** prod Δ **empty** @ T-IGX-066-A mechanical replay — `historical_evidence_invalidated_by_later_changes` **0**. Historical PASS records @ older TEST_EXECUTION_SHA remain archival; Wave 06 ECAP counter table refreshed @ implementing commit (ticket checkpoint).

## C3 — VBN taxonomy (31 domains)

Seven replay artifacts @ normative schema (§2.1–2.7) + EBH-5/6 historical SHA citations.

## C4 — Drift vs accepted baselines

[`ROADMAP_REPLAY_X_PROD_QUAL_DRIFT_LOG.md`](ROADMAP_REPLAY_X_PROD_QUAL_DRIFT_LOG.md) — scoped prod path recorded; no parent reopen from **unscoped** drift.

## C5 — Qualification replay plan

[`ROADMAP_REPLAY_X_QUALIFICATION_MANIFEST.md`](ROADMAP_REPLAY_X_QUALIFICATION_MANIFEST.md) — **L1/L2/L3 mechanical PASS** @ `TEST_EXECUTION_SHA` **`22fbc0e837e819fe2a068295df5f527612dcf307`** (logs `.tmp/proof/roadmap-replay-x-immutable-6fbc4a8/`, **EVIDENCE_EXPORT_REQUIRED**).

## Six counters (mechanical @ `TEST_EXECUTION_SHA`)

Computed by `tests/qualification/roadmap_replay_x/_closure_counters.py` + B6 R2 bidirectional reconciliation @ **T-IGX-066-A** mechanical replay (evidence pin `da5ca5b53`; implementing commit SHA in `governance/tickets/T-IGX-066-A.md` checkpoint).

| Counter | Value | Denominator / source |
| --- | ---: | --- |
| reopened_historical_stages | 0 | §5 category A rows + `_formal_reopen_register.py` |
| unresolved_replay_blockers | 0 | `_replay_blocker_register.py` |
| roadmap_drift | 0 | `_roadmap_drift.py` |
| historical_evidence_invalidated_by_later_changes | 0 | unscoped prod git diff + scoped disposition validation |
| unclassified_replay_findings | 0 | §5 rows + ECAP/VBN/arch/qual inventories (`_replay_finding_ledger.py`) |
| vendor_neutrality_replay_failures | 0 | 7 VBN artifacts + `ebh_6/_findings_register.py` @ HEAD |

## External gates (closed @ W10)

| Gate | Status |
| --- | --- |
| B6 Protocol 03 | **ISSUED — PASS (SCOPED)** |
| Operator OD | **APPROVED — OD-REPLAYX-08** |
| L2 parent replay @ TEST_EXECUTION_SHA | **PASS (SCOPED)** |
| L3 ECAP bundle @ TEST_EXECUTION_SHA | **PASS** |
| EBH-7 | **NOT ENTERED** (eligible successor) |
