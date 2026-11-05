# IIAC governance implementation — Session A terminal report

**Session:** IIAC-GOVERNANCE-REFORM Session A  
**Ticket:** [`governance/tickets/T-IGX-016-A.md`](../../../../governance/tickets/T-IGX-016-A.md)  
**Branch:** `development`  
**Baseline @ Session A start:** `eab65e010f5a1abfc413bb05419435bdf6b1810b`  
**Policy status:** **NOT_IN_FORCE** (by design — no Session B PASS)

## Disposition

**IMPLEMENTED** — G0+G1+G2+G3 deliverables on branch; activation deferred to Session B.

## SHAs

| Label | Value |
|-------|-------|
| `GOVERNANCE_IMPLEMENTATION_SHA` (Session B audit pin; unchanged) | `2ae1014116d353bc7742985e62f2bccc2bfac798` |
| `PUBLICATION_HEAD` (`development` after Session A doc reconciliation) | `4bb192889fea9d8883f352384fcb4cb48d04601b` |
| `IIAC_POLICY_STATUS` | **NOT_IN_FORCE** |

## Test results

| Command | Result |
|---------|--------|
| `uv run pytest tests/unit/governance/test_iiac_negative_guards.py -q` | **13 passed** (log: `.tmp/session/IIAC-GOV-A/pytest-iiac.log`) |
| `verify_iiac_policy_activation.py --expect-not-in-force` | **PASS** (NOT_IN_FORCE) |
| `verify_iiac_audit_manifest.py` (example manifest) | **PASS** |
| `verify_iiac_role_separation.py` (example manifest) | **PASS** |
| `verify_od_ent_02_compensating_control.py` | **PASS** (draft package) |
| `evol_x_closure_publication_preflight.py` | **FAIL** (expected — legacy B6 authority) |
| `ent_audit_x_formal_closure_preflight.py` | **FAIL** (expected — legacy B6/OD/FRZ) |
| `uv run python -m scripts.governance.igx_gov ticket` | **OK** (log: `.tmp/session/IIAC-GOV-A/igx-gov-ticket.log`) |

## G0

- [`scripts/governance/generate_iiac_authority_impact_matrix.py`](../../../../scripts/governance/generate_iiac_authority_impact_matrix.py) + [`iiac_g0_inventory.py`](../../../../scripts/governance/iiac_g0_inventory.py) — governed-path enumeration (v2 matrix)
- [`verify_iiac_g0_inventory_completeness.py`](../../../../scripts/governance/verify_iiac_g0_inventory_completeness.py)
- [`IIAC_GLOBAL_AUTHORITY_IMPACT_MATRIX.json`](./IIAC_GLOBAL_AUTHORITY_IMPACT_MATRIX.json) — see F01 corrective addendum below

## F01 corrective addendum (Session A batch)

| Field | Value |
|-------|-------|
| Session B disposition addressed | F01 (blocker), F02, F03 |
| `GOVERNANCE_IMPLEMENTATION_SHA` (historical) | `2ae1014116d353bc7742985e62f2bccc2bfac798` (unchanged) |
| **`CORRECTIVE_IMPLEMENTATION_SHA`** (publication HEAD @ push) | `7d6061e48c6f84bbf1b05e34a28d4fd495602ae8` |
| Matrix publication commit | `9bd8559b36948aefd8134232b89b7738d6dd5bbc` |
| `INVENTORY_SOURCE_SHA` | `afe9e8abeabdbab410bf562ebed6f4a309a91057` |
| `inventory_publication_sha` | `9bd8559b36948aefd8134232b89b7738d6dd5bbc` |
| G0 v2 stats | **17510** covered paths · **18388** entries · schema `iiac-authority-impact.v2` |
| Handoff (F03) | [`IIAC_SESSION_HANDOFF_RECORD_2026_11_05_SESSION_A_B.md`](./IIAC_SESSION_HANDOFF_RECORD_2026_11_05_SESSION_A_B.md) — operator line **pending** |

**Prior G0 @ Session A close:** 3492 pattern-only entries @ `eab65e0…` — superseded by v2 full enumeration after F01 fix.

## G1 / G2 / G3 (summary)

- IIAC protocol, migration matrix, activation JSON template, manifest schema, handoff template, OD-ENT-02 threat model + compensating package
- Validators under `scripts/governance/verify_iiac_*` + `verify_od_ent_02_compensating_control.py`
- ECAP, FRZ, roadmap, ENT/EVOL matrix, DevOS handoff, `.cursor/rules/intergrax-independent-audit.mdc`
- Session B/C/D instructions + certification publication contract
- Example manifest (non-PASS): [`examples/iiac_audit_session_manifest.example.json`](./examples/iiac_audit_session_manifest.example.json)

## Blockers (none for Session A close)

- Session B independent governance audit required before `IN_FORCE`
- EVOL/ENT formal preflights remain on legacy fail-closed until activation

## Handoff

Session B: [`IIAC_SESSION_B_INDEPENDENT_GOVERNANCE_AUDITOR_INSTRUCTIONS.md`](./IIAC_SESSION_B_INDEPENDENT_GOVERNANCE_AUDITOR_INSTRUCTIONS.md) @ pinned `GOVERNANCE_IMPLEMENTATION_SHA`.
