# IIAC ACT-01 corrective dossier

**Policy:** `IIAC-2026-11-05.1`  
**Parent baseline:** `5d6b44b05f90ecf420d45d01659ea64565b52848`  
**Historical governance SHA:** `2ae1014116d353bc7742985e62f2bccc2bfac798`  
**Re-audit PASS digest:** `62b93665a97024337638e966ca87cbe37f32e1b4c9c795bef46e4c1097aefbde`  
**Historical PARTIAL digest:** `56c9b078e42e1408799e37a2535aa61879e31021864214e46dfa165fb41fded1`  
**Canonical status:** `NOT_IN_FORCE`  
**This dossier is not a PASS and not an operator signature.**

## Root cause

`verify_iiac_policy_activation.py` returned exit 0 for `NOT_IN_FORCE` without `--expect-not-in-force`. For `IN_FORCE` it accepted any nonempty audit digest, any nonempty manifest string, and any 40-character ancestor SHA. It did not read artifact bytes, require verdict `PASS`, require an operator decision, or run the manifest and role-separation scripts.

## Correction

Default and `--require-in-force` exit 0 only for a fully bound `IN_FORCE` record. `--expect-not-in-force` exits 0 only when the record is inactive and carries no active claim. Evidence bytes are hashed with SHA-256. Historical `historical_session_b/` cannot be the active manifest. `effective_from_sha` must be a strict ancestor whose blob is already `IN_FORCE` with an empty effective SHA (phase 1). Phase 2 records that commit. The gate does not close EVOL-X, ENT-AUDIT-X, or ARCH-FREEZE.

## Qualification

`uv run pytest tests/unit/governance/test_iiac_activation_gate.py tests/unit/governance/test_iiac_negative_guards.py tests/unit/governance/test_iiac_g0_inventory.py -q`

Result: **45 passed, 1 skipped** in 116.83s. The skip is symlink creation on this Windows host. ACT-N14 path traversal was executed and rejected.

| Check | Result |
|---|---|
| `--expect-not-in-force` on canonical record | exit 0, `NOT_IN_FORCE` |
| default and `--require-in-force` on canonical record | exit 1, `ACT-N01` |
| G0 completeness | PASS, 17510 paths @ `afe9e8ab…` |
| OD-ENT-02 package | PASS (package draft) |
| example manifest and role separation | PASS |
| EVOL-X and ENT-AUDIT-X closure preflights | exit 1, unchanged fail-closed (missing legacy B6 acceptance). Not promoted to CLOSED. |

## Preserved evidence

Copies under `docs/project/maintainers/qualification/iiac_evidence/IIAC-GOV-1/` keep the original report digests. The G0 matrix at the parent baseline was left in place (`inventory_source_sha` `afe9e8ab`, 17510 paths).

## Not done

No canonical `IN_FORCE`. No signed `IIAC_OD_ACTIVATION_*.md`. No self-certification. Auditor handoff: `IIAC_ACT01_INDEPENDENT_REAUDIT_HANDOFF.md`.
