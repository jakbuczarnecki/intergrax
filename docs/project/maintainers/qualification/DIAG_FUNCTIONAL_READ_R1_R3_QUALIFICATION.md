# DIAG-FUNCTIONAL-READ-R1-R3 Qualification

**Verdict:** PASS

**Date:** 2026-09-04

**Branch:** `development`

**Start HEAD:** `0366e855556cc165a481d901526d0f72b2e21233`

**R1-R2 qualified SHA:** `46159f15b8f82642e09ae67a04478e31003cdac2`

**R1-R3 qualified SHA:** _(commit SHA below)_

## Defect closed

Independent audit: R1-R2 reader could delete `appendpending` when canonical was missing, classifying it as orphan. Active writer paused before canonical → reader deleted intent → writer wrote canonical → crash before v2/v1 → silent canonical omission possible.

## Architecture options

| Option | Summary | Verdict |
| ------ | ------- | ------- |
| A — fail closed on pending + no canonical | Query raises `FunctionalEvidenceProjectionConsistencyPendingError`; never delete intent during read | **Selected** |
| B — lease-based append intent | `owner_token` + `lease_until` + generation fencing | Rejected (clock/lease scope) |
| C — explicit recovery claim | `PENDING → RECOVERY_CLAIMED` with writer fencing | Rejected (heavier than required) |

**Selected:** Option A — `canonical missing != writer abandoned`.

## Error contract

| State | Error |
| ----- | ----- |
| pending + canonical missing | `FunctionalEvidenceProjectionConsistencyPendingError` |
| pending + canonical present | repair v2/v1; clear intent |
| pending corrupt / key mismatch | `FunctionalEvidencePersistenceIntegrityError` |
| pending + conflicting index | `FunctionalEvidencePersistenceIntegrityError` |

## Query rule

```text
for each pending:
    if canonical exists: repair projections; clear intent
    else: DO NOT CLEAR; raise consistency-pending
```

## Active vs abandoned

Under Option A both behave identically (fail closed). The system does not guess abandonment — explicit writer retry or maintenance resolves abandoned intents.

## Unit proofs

`tests/unit/runtime/diagnostics/test_diag_functional_read_r1_r3_active_writer_safety.py` — 17 cases including deterministic active-writer/reader race with `threading.Event`.

## Process restart proof (real Mongo)

Profile:

- E = 1000 healthy evidence
- PROCESS A: seed + create pending only (simulates writer before canonical)
- PROCESS B: consistency-pending reader subprocess — must fail closed, pending durable
- PROCESS A/B: writer retry with fault after canonical
- PROCESS C: recovery reader subprocess — must recover E+1

Artifact: `.tmp/proof/diag-functional-read-r1r3/mongo-active-writer-qualification.json`

| Metric | Value |
| ------ | ----- |
| base_evidence_count | 1000 |
| reader_fail_closed | true |
| pending_after_reader | true |
| expected_count | 1001 |
| recovered_count | 1001 |
| passed | **true** |

## Files changed

- `intergrax/runtime/diagnostics/functional_evidence_persistence.py`
- `intergrax/runtime/diagnostics/functional_evidence_projection_repairer.py`
- `tests/unit/runtime/diagnostics/test_diag_functional_read_r1_r2_append_crash_safety.py`
- `tests/unit/runtime/diagnostics/test_diag_functional_read_r1_r3_active_writer_safety.py` (new)
- `tests/system/functional_diagnostics_read_r1_r3/` (new)
- `tests/system/functional_diagnostics_read_r1_r2/mongo_recovery_qualification.py`
- `docs/project/architecture/DIAGNOSTICS.md`
- `docs/project/maintainers/qualification/DIAG_FUNCTIONAL_READ_R1_R2_QUALIFICATION.md`

## Final architecture statement

```text
ACTIVE APPEND INTENT = NEVER RECLAIMED BY READER WITHOUT PROOF
PENDING + CANONICAL MISSING = CONSISTENCY PENDING / FAIL CLOSED
PENDING + CANONICAL PRESENT = DETERMINISTIC REPAIR
CANONICAL FUNCTIONAL EVIDENCE = SOLE SOURCE OF TRUTH
SILENT CANONICAL OMISSION = IMPOSSIBLE WITHIN QUALIFIED MODEL
```

## Remaining limits

Abandoned pending without canonical blocks query until explicit retry/maintenance — intentional; automatic garbage collection deferred.
