# EBH-7 — Behavioral remediation backlog (completion wave)

**Baseline HEAD:** `e2467a88ed042e2d97150568f9f6b25ba85cb6b1`  
**Gate SSOT:** `tests/qualification/ebh_7/`  
**Verdict class:** PARTIAL — exact blockers below

## RPL-H (Phase B)

| Metric | @ HEAD |
| --- | --- |
| QUALIFIED | 5 |
| NOT YET (behavioral OPEN) | 51 |
| Exit | Per-concern EBH-5 structural + behavioral + negative proofs; batch by shared port/owner |

Historical baseline `51 NOT YET / 5 QUALIFIED` is **not** a permanent gate fixture — acceptance tracks **current** `CROSS_LAYER_REPLACEABILITY_MATRIX` classifications only (`assert_ebh7_rpl_h_disposition_integrity`).

## Tenant E2E (Phase C) — 12 OPEN

| ID | Disposition | Removal condition |
| --- | --- | --- |
| TEN-E2E-03 | PARTIAL | Gov→Exec cross-admission deny on declared path |
| TEN-E2E-04 | BLOCKED | Canonical Exec→Tool tenant isolation proof (no shadow resolver) |
| TEN-E2E-05 | PARTIAL | Integration factory boundary without broker-only approximation |
| TEN-E2E-12–14, 17–19, 21–23 | MISSING | Register + execute platform E2E proof nodes @ HEAD |
| TEN-E2E-13 | PARTIAL | Plugin host activation × tenant (not COMPAT-only probe) |

**EXACT (14):** 01, 02, 06–11, 15, 16, 20, 24–26 — see `EBH_7_SEMANTIC_PROOF_BINDING_AUDIT.md`.

## Cross-domain (Phase D) — 18 OPEN

| Disposition | Count | IDs |
| --- | --- | --- |
| MISSING_PROOF | 16 | 007, 012, 014–015, 017, 019, 022–031 |
| STRUCTURAL_ONLY | 1 | 016 |
| PARTIAL_PROOF | 1 | 018 |

**EXACT (16):** remaining mapped scenarios with `EXACT_PROOF` in semantic audit.

## Enterprise closure (unchanged)

- Protocol 03 **B6:** OPEN — no agent self-approval  
- **OD-EBH7-01 / OD-EBH7-08:** OPEN  
- No global FRZ promotion; Path B vendor live qualification separate

## Next batch order (operator-approved)

1. TEN-E2E-04 architecture unblock or formal BLOCKED record with remediation owner  
2. High-risk XDM: 007, 012, 014–015 (Tool×Skill×Plugin, Trace×Replay, Cancel×Effects)  
3. RPL-H batches grouped by `integration.*` store/bus ports (EBH-5 behavioral suite)  
4. Re-run `uv run pytest tests/qualification/ebh_7/ -q` → publish `EBH_7_TEST_EXECUTION_RECORD.md` @ commit SHA
