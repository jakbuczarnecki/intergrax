# GR-13 — Governance Evidence Proof Matrix

**Baseline SHA:** `4e5878f538521ef2bc7e080599574e23d8821a5a`  
**Scope authority:** `docs/project/technical/adr/entries/2026-09-21/ADR-GR-10-003-gr10-gr13-governance-evidence-certification-scope.md`  
**Deferred inventory SSOT:** `GR13_AGENTIC_GOVERNANCE_EVIDENCE_DEFERRED` / `GR13_ORCHESTRATION_GOVERNANCE_EVIDENCE_DEFERRED` in `tests/qualification/governance/strategy/catalog.py`  
**Mechanical matrix:** `tests/qualification/governance/gr13/catalog.py` + gates `tests/qualification/governance/gr13/test_gr13_governance_evidence_proof_matrix.py`

## Canonical contracts

- Fact: `intergrax.contracts.governed_execution_governance_evidence.GovernanceDecisionEvidenceFact`
- Persistence: `GovernanceEvidencePersistencePort`
- Recorder: `intergrax.runtime.governance.governance_evidence_recorder.GovernanceEvidenceRecorder`
- Shared projection helper: `intergrax.runtime.governance.governance_policy_decision_evidence_recording`

## GR-10 / GR-11 / GR-12 reconciliation

| Item | Status |
|------|--------|
| GR-10 | FINAL CLOSED within GR-10 scope; per-GEP evidence deferred rows consumed by GR-13 |
| GR-11 | READY FOR AUDIT @ `4e5878f538521ef2bc7e080599574e23d8821a5a` |
| GR-12 | Independently accepted @ `03dde6c68a37ac0a8fe19cc5bcf683da8a3afc06` |

## Scenario Y (GOV-FINAL-4)

Reconciled to **QUALIFIED** using existing host recovery integrity proofs (`test_gr7_a4_unknown_host_state_separation.py`, `applications/governed_contractor_application/tests/host/test_gr7_a7_r1_recovery_integrity.py`) — no new recovery architecture under GR-13.

## Scoped FRZ contribution (this task only)

Evidence toward forensic traceability of governance decisions at deferred GEPs: FRZ-GOV-07, FRZ-GOV-08, FRZ-TRC-01, FRZ-TRC-03, FRZ-TRC-07, FRZ-TRC-10 (scoped; not global PASS).

## Tests

```text
uv run --frozen pytest tests/qualification/governance/gr13/ -q -p no:xdist
uv run --frozen pytest tests/unit/runtime/governance/test_gr13_governance_evidence_gep_emission.py -q -p no:xdist
uv run --frozen pytest tests/qualification/governance/ -q -p no:xdist  # ×2
```

## Recommended status

**GR-13-R1 = READY FOR AUDIT** (pending independent exact-SHA GitHub audit)
