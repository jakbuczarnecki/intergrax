# EQO EDER — 1073-row disposition (same register)

**Parent:** `EQO-E` · **Register:** `enterprise_debt_register/register.jsonl` (no second register).

## Row accounting (E-01)

```
1073 = 1069 (reconciliation ingest) + EDER-N-0001..N-0004
```

Update `reconciliation.json` `rows_total` to **1073** via docs ticket (046/047 or steward vehicle from W0-02). If no ticket: one `WAIT_FOR_OPERATOR` row, not a forked register.

## E-02 — Mechanical rules (batched execution)

Run as **one** steward pass; do not ask Operator per row.

| Rule | Action |
|------|--------|
| R1 UNRATED + finding has severity | Copy severity; `severity_source=FINDING` |
| R2 `next_action` map | DOC_FIX→DOC_PATCH; CAMPAIGN_B/C→VERIFY_TEST; REQUALIFY→REQUALIFY; OWNER_DECISION→DECISION; NONE→hold for W4-05 |
| R3 path bytes `A_BASELINE` vs `DOC_PIN` / `TRAIN_PIN` | UNCHANGED or CHANGED → re-verify queue |
| R4 duplicates `linked_finding_ids` / `origin.id` | SUPERSEDED only with surviving proof; else `DUPLICATE_UNRESOLVED` |

**Acceptance:** publish counts — remaining UNRATED (only missing source severity), remaining UNCLASSIFIED (only missing mapping).

## E-03 — Owner batches (OD-EQO-09)

| Bucket | Count (plan) | Mechanism |
|--------|--------------|-----------|
| SINGLE_PROPOSED vs DP-47 | — | one confirm decision |
| MISSING | 15 | list in OD |
| UNMAPPED | 14 | list in OD |
| SHARED_UNRESOLVED | 75 | list in OD |
| ROLE_NOT_OWNER | 12 | list in OD |
| GENERIC_NOT_AN_OWNER | 4 | list in OD |

Rows stay `INGESTED` until TRIAGED; EQO matrix disposition = `WAIT_FOR_OPERATOR`.

## E-04 — `needs_independent_audit=yes` (147)

No 147 separate audits: bind to WB `AUD` / C `AUD` — one verdict covers the workstream’s finding set. Each row gets `independent_audit_ref` or `tbd` + owning workstream.

## E-05 — Operator package (OD-EQO-10)

Only when: new `EXCEPTION_ACCEPTED`, impact → `BLOCKS`, or `accountable_person` change. Existing 155+39 accepted rows stand unless E-02 shows CHANGED. **UNKNOWN impact cannot become exception** (README rule).

## E-CLS

Every one of 1073 rows has EQO disposition class: mechanical, owner batch, audit ref, or Operator package. Lifecycle may remain `INGESTED` if TRIAGED gate not met and reason is on E-03/E-05 list.

## Dependency

`EQO-W0-02` → E-01; `EQO-E-02` parallel to W3; E-CLS on critical path for BLOCKS/UNKNOWN (plan §5).