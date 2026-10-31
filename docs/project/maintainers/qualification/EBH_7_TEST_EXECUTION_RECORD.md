# EBH-7 — Test execution record

| Field | Value |
| --- | --- |
| Closure wave (superseded for semantic) | `f913944ff608c97a4a4145aae6c2bbd610004f64` |
| W1/W2 reconciliation | baseline `5df074fcb` → recon `9ecf8666a` |
| Batch A/B (OD-EBH7-01 full execution) | start `195046d46` → impl `ab8f958ed` |
| Batch C RPL-H | start `da1c29162` → impl `473462bae` |
| Batch D TEN/XDM | start `4c884dbc5` — D1 TEN-03/04/05 EXACT promotion |
| Command | `uv run pytest tests/qualification/ebh_7/ -q` @ Batch C HEAD |
| Result | **120 passed, 4 skipped, 2 xfailed** (B6 gates; TEN-13 harness skip) |
| Log | `.tmp/session/ebh-7-batch-c/pytest-ebh7-full.log` (capture on next run) · Batch C subset: `.tmp/session/ebh-7-batch-c/pytest-batch-c.log` |

## L1 / L2 / L3 (@ semantic correction)

| Layer | Result |
| --- | --- |
| L1 | PASS — disposition SSOT complete; MD sync gate; no auto-EXACT |
| L2 | PASS — EXACT-scenario proof execution + Package C mechanical |
| L3 | PASS — full `tests/qualification/ebh_7/` with documented XFAIL |

## Semantic verdict (agent)

**`EBH-7 — SEMANTIC PROOF INTEGRITY PARTIAL / EXACT BLOCKERS`**

- TEN: **17 EXACT / 9 PARTIAL** (+3 Batch D D1 @ disposition SSOT)
- XDM: **16 EXACT / 18 OPEN**
- RPL-H semantic: **6 QUALIFIED / 29 PARTIAL / 21 BLOCKED** — SSOT `tests/qualification/ebh_7/_rpl_h_semantic_evidence.py` (no static allowlist promotion)

## FRZ impact (scoped — no status change)

- **FRZ-TEN-***: ECAP C4/C5 counts revised downward; tenant E2E claims require disposition EXACT, not registry-only proof nodes.
- **Global FRZ PASS**: not claimed.
- **FRZ-PRD / live slice**: unchanged OPEN (PROD-Q delegated EVOL).

## Full stage closure checklist

| Check | Status |
| --- | --- |
| Fail-closed semantic audit | PASS |
| 60/60 EXACT_PROOF | **OPEN** (33/60 EXACT @ Batch D partial) |
| RPL semantic QUALIFIED 56/56 | **OPEN** (6/56 QUALIFIED; 21 BLOCKED w/ root cause) |
| OD-EBH7-01 | **APPROVED** @ `81679e672` — operator | B6 / OD-EBH7-08 remain **OPEN** |
| Protocol 03 B6 | OPEN — independent auditor |
| OD-EBH7-08 | OPEN — operator |
| Roadmap EBH-7 CLOSED | **not claimed** |
