# GR-13 — Governance Evidence Proof Matrix (R2)

**Task baseline SHA:** `01fe7ae352d0761f1b443448fdd6933cde76d436`
**R2 qualification commit:** `dd8234ad794271947b65cd9dae9419208e6ccf22`
**Scope authority:** `docs/project/technical/adr/entries/2026-09-21/ADR-GR-10-003-gr10-gr13-governance-evidence-certification-scope.md`  
**Deferred inventory SSOT:** `GR13_AGENTIC_GOVERNANCE_EVIDENCE_DEFERRED` / `GR13_ORCHESTRATION_GOVERNANCE_EVIDENCE_DEFERRED` in `tests/qualification/governance/strategy/catalog.py`  
**Mechanical matrix:** `tests/qualification/governance/gr13/catalog.py` + gates `tests/qualification/governance/gr13/test_gr13_governance_evidence_proof_matrix.py`

## R2 proof model

- Twelve deferred rows (7 AGENTIC + 5 ORCHESTRATION); no second applicability SSOT.
- No self-declared `QUALIFIED` on matrix rows — gates G13-02..G13-14 establish qualification from mechanical proof.
- Positive proof must enter through the **canonical production owner/path** named by GR-10 semantics (markers in `canonical_path_markers` per row).
- Helper-only proof (`record_governance_policy_decision_evidence*`, `record_tool_plan_or_access_evidence`) cannot qualify a row (G13-02).
- Shared GEPs (`TOOL_PLAN_OR_ACCESS`, `TOOL_INVOCATION_POLICY`, `POST_RUN`) use `SHARED_CANONICAL_PATH`; distinct strategies still have separate pytest nodes exercising the same emission module.

## Canonical contracts

- Fact: `intergrax.contracts.governed_execution_governance_evidence.GovernanceDecisionEvidenceFact`
- Persistence: `GovernanceEvidencePersistencePort`
- Recorder: `intergrax.runtime.governance.governance_evidence_recorder.GovernanceEvidenceRecorder`
- Shared projection helper (not a positive-proof entry): `intergrax.runtime.governance.governance_policy_decision_evidence_recording`

## GR-10 / GR-11 / GR-12 reconciliation

| Item | Status |
|------|--------|
| GR-10 | FINAL CLOSED within GR-10 scope; per-GEP evidence deferred rows consumed by GR-13 |
| GR-11 | CLOSED / ACCEPTED @ `4e5878f538521ef2bc7e080599574e23d8821a5a` |
| GR-12 | CLOSED / ACCEPTED @ `03dde6c68a37ac0a8fe19cc5bcf683da8a3afc06` |

## Scenario Y (GOV-FINAL-4)

Mechanically bound via G13-14 to `GR13_SCENARIO_Y_PROOF_NODES` (same nodes as `GOV_FINAL_4_SCENARIO_CATALOG` scenario **Y**):

- `test_gr7_a4_unknown_host_state_separation.py::test_crash_ambiguity_intent_without_outcome_not_explicit_unknown`
- `test_gr7_a7_r1_recovery_integrity.py::test_crash_ambiguity_forged_repeat_blocked`

## G13-01..G13-14 (mechanical gates)

| Gate | Subject |
|------|---------|
| G13-01 | Closed-world inventory vs GR-10 deferred |
| G13-02 | Required rows proven via canonical path markers (not declarative QUALIFIED) |
| G13-03 | Canonical fact contract on all rows |
| G13-04 | Canonical persistence contract on all rows |
| G13-05 | Owner, production path, emission module integrity |
| G13-06 | Shared canonical path convergence for shared GEPs |
| G13-07 | Negative decision / authority proof nodes declared |
| G13-08 | Evidence non-authoritative nodes registered |
| G13-09 | Identity correlation helper used in unit proofs |
| G13-10 | No duplicate fact/persistence classes in registered emission modules |
| G13-11 | GR-8 evidence failure nodes collectable |
| G13-12 | All proof nodes collectable |
| G13-13 | GR-11 / GR-12 accepted SHAs reconciled |
| G13-14 | Scenario Y QUALIFIED bound to recovery proof nodes |

## Tests

```text
uv run --frozen pytest tests/qualification/governance/gr13/ -q -p no:xdist
uv run --frozen pytest tests/unit/runtime/governance/test_gr13_governance_evidence_gep_emission.py -q -p no:xdist
uv run --frozen pytest tests/qualification/governance/ -q -p no:xdist  # ×2
```

Latest verification (R2): 31 GR-13 scoped tests; 394 governance qualification tests ×2 (identical pass count).

## Recommended status

**GR-13-R2 = READY FOR AUDIT** (pending independent exact-SHA GitHub audit)
**GR-13 = READY FOR AUDIT**
**GOV-X1 = CURRENT** (not global FRZ closure)
