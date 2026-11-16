# PLAN T-IGX-036-D — PDC-P3 ownership and decision drafts (architecture lock preparation)

Parent: PDC-X master plan [`PLAN_T-IGX-036-A.md`](PLAN_T-IGX-036-A.md) §F and §G "PDC-P3". Depends on T-IGX-036-C (P2). No product code.

## Deliverables

1. **PDC-P3-01:** `PDC_OWNERSHIP_MATRIX.md`, recording exactly one semantic owner and composition root per D1–D8 concern. It resolves the dual-owner findings from P2: D7-R022 (task registries), D8-R017 and D8-R018 (artifact contracts and the name collision), and D2-R017 (`ContextFragment`). Each resolution is checked against the recorded DUP-X dispositions.
2. **PDC-P3-02:** ADRs, all with status **Proposed**:
   - `docs/project/maintainers/architecture/ADR_PDC_01_AGENT_UPGRADE_RETIRE.md`
   - `ADR_PDC_03_MODEL_STREAMING_CAPABILITY_CONTRACT.md`
   - `ADR_PDC_04_HOSTING_LIFECYCLE_STABLE_API.md`
   - `ADR_PDC_05_EVALUATION_CONTRACTS.md`
   - `ADR_PDC_06_RUNTIME_SECURITY_HARDENING.md`
   - `ADR_PDC_07_BACKGROUND_RECURRENCE_AND_DLQ.md`
   - `ADR_PDC_08_ARTIFACT_LIFECYCLE_TAXONOMY.md`

   Each ADR cites the gap IDs it covers and the canonical mechanism it reuses, and explicitly rejects any new parallel authority. The ADR set is indexed in `PDC_ARCHITECTURE_DECISIONS.md`.
3. **PDC-P3-03 and PDC-P3-05:** operator decision drafts, all **PENDING**, each with a recommended option and evidence:
   - OD-PDC-SCOPE-01: SKILL-X F-04
   - OD-PDC-SCOPE-02: APP-HOST-6/7/9E
   - OD-PDC-SCOPE-03: LLM provider plugin / OCI factory
   - OD-PDC-SCOPE-04: D8 secure download
   - OD-PDC-SCOPE-05: D7 recurrence
   - OD-PDC-PATHB-01
   - OD-PDC-ADR-01: acceptance of the ADR set
4. **PDC-P3-04:** OD-PDC-IIAC-SCOPE is refined with the auditor and certifier session plan. It stays pending.
5. **PDC-P3-06:** the artifact feature doc pair (Principles §14) is specified inside ADR-PDC-08. The doc pair itself is created only after OD-PDC-ADR-01 is accepted.
6. **PDC-P3-07:** ECAP §16 impact decision recorded in `PDC_ARCHITECTURE_DECISIONS.md`.
7. The operator decision register is updated.

## Acceptance

- No dual owner is left undecided.
- Every deferral candidate is an explicit OD draft; nothing is silently deferred.
- No ADR is marked Accepted and no OD is marked signed. The architecture lock (PDC-P3-08) remains an Operator gate.

## Test contract

```text
uv run python scripts/qualification/pdc_x_requirements.py --check
uv run python scripts/docs/check_docs_domain_pairs.py
uv run python -m scripts.governance.igx_gov ticket
```

## Out of scope

Signing any OD; ADR acceptance; product code; new doc pairs before ADR acceptance; P4 design.
