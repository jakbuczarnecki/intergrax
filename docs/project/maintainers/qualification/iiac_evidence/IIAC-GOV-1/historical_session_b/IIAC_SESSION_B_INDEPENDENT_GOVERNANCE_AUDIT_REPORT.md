# IIAC Session B — Independent governance audit report

**Role:** `independent-quality-auditor`  
**Stage:** `IIAC-GOV-1`  
**Policy version:** `IIAC-2026-11-05.1`  
**Audited implementation SHA (`GOVERNANCE_IMPLEMENTATION_SHA`):** `2ae1014116d353bc7742985e62f2bccc2bfac798`  
**Audit execution SHA:** `82f5fa319d1f33c453d9aa506d6e042e5839dcc2`  
**Branch:** `development`  
**Ancestry:** `merge-base --is-ancestor` audited → audit execution **OK**

## Normative scope reviewed

- [`INTEGRAX_INTERNAL_INDEPENDENT_AGENT_CERTIFICATION_PROTOCOL.md`](../../../docs/project/maintainers/qualification/INTEGRAX_INTERNAL_INDEPENDENT_AGENT_CERTIFICATION_PROTOCOL.md)
- [`ENTERPRISE_CONTINUOUS_ARCHITECTURE_ASSURANCE_PROTOCOL.md`](../../../docs/project/maintainers/qualification/ENTERPRISE_CONTINUOUS_ARCHITECTURE_ASSURANCE_PROTOCOL.md) (IIAC transition / ECAP-P4)
- [`PLATFORM_ENTERPRISE_COMPLETION_ROADMAP.md`](../../../docs/project/maintainers/plans/PLATFORM_ENTERPRISE_COMPLETION_ROADMAP.md) (EVOL/ENT status vs IIAC NOT_IN_FORCE)
- [`PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST.md`](../../../docs/project/maintainers/qualification/PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST.md)
- [`ENT_EVOL_CROSS_STAGE_DECISION_MATRIX.md`](../../../docs/project/maintainers/qualification/ENT_EVOL_CROSS_STAGE_DECISION_MATRIX.md)
- Session A diff @ pinned SHA (32 files, G0–G3)
- Post-pin delta `2ae10141..82f5fa3`: **docs/ticket only** (no `intergrax/`, `agents/`, `applications/`)

## Independent replay (executed @ audit execution SHA)

| Check | Result | Evidence digest (SHA-256) |
|-------|--------|---------------------------|
| `pytest tests/unit/governance/test_iiac_negative_guards.py -q` | **13 passed** | `a0fb7c880d53ab4f3677ef882237cb2ea1dec30b6db4c67dea5131a66dcffc9d` |
| `verify_od_ent_02_compensating_control.py` | **PASS (draft)** | `506f6a3aaa370715f0f5ce8de8892ebc0163f66b830c144e8a7e6edfb7519a68` |
| `verify_iiac_policy_activation.py --expect-not-in-force` | **PASS** | `7b15055e8ce16ed757acaa35129f3738d3eb127b491a08a403304bb46dfc5b55` |
| `evol_x_closure_publication_preflight.py` | **FAIL (expected)** | legacy B6 fail-closed |
| `ent_audit_x_formal_closure_preflight.py` | **FAIL (expected)** | legacy B6/FRZ fail-closed |
| Adversarial N-09 | **PASS** | no `*IIAC_INDEPENDENT_AUDIT_PASS_RECORD.md` in qualification tree |
| `IIAC_POLICY_ACTIVATION_RECORD.json` | **NOT_IN_FORCE** | not modified in this session |

## G0 authority-impact inventory (full structural review)

- **entry_count vs entries:** 3492 = 3492 (**OK**)
- **Category totals:** {1:716, 2:115, 3:976, 4:360, 5:685, 6:204, 7:436}
- **Category 1 path existence:** 0 missing files on disk
- **Matrix file digest @ audit execution:** `f3c4a83d18e6ed3e7d1beb4abb3f934a44fad5d3ac0c4cc770ac1b57c98c6310`
- **Regeneration @ HEAD:** semantic inventory **≠** committed matrix (`repository_head_at_generation` still `eab65e0…`)
- **Coverage gap:** **≥18** Session A authority-bearing paths (including IIAC protocol, session B/C/D instructions, validators, `T-IGX-016-A.md`) **absent** from matrix because generation ran @ Session A baseline **before** those files existed and was **not** re-run @ `GOVERNANCE_IMPLEMENTATION_SHA`

## ECAP Mode C — category 1 spot checks (sample)

Deterministic sample (seed 42): five category-1 paths verified **present** on disk; narrative in EVOL/ENT roadmap rows distinguishes **internal** closure from **formal Protocol 03 / IIAC** closure (**no false global CLOSED** observed in amended roadmap IIAC lines).

## Adversarial negatives (Session B)

| Threat | Result |
|--------|--------|
| Self-certification / premature PASS files | **Not observed** |
| Unauthorized `IN_FORCE` | **Blocked** (N-08, activation record clean) |
| Stale audited SHA | Post-pin changes **non-product**; strict role-separation **PASS** for manifest |
| Self-merge auditor=implementer session | Enforced by validator (N-01) |
| OD-ENT-02 equivalence claim | **Absent** (N-06) |

## Findings register

| ID | Sev | Root cause | Evidence | Corrective owner | Fix SHA |
|----|-----|------------|----------|------------------|---------|
| **IIAC-GOV-B-F01** | **BLOCKER** | G0 matrix generated @ `eab65e0…` and shipped without regeneration after Session A artifacts landed @ `2ae10141…`; normative IIAC/governance paths missing from inventory | `repository_head_at_generation`; path grep shows zero hits for `INTEGRAX_INTERNAL…`, `verify_iiac_*`, `T-IGX-016-A.md`; 18 new text paths from Session A diff not in matrix path set | **implementer** (Session A follow-up / T-IGX-016-A) | **pending** — target regen @ `GOVERNANCE_IMPLEMENTATION_SHA` or successor doc commit |
| **IIAC-GOV-B-F02** | **LOW** | Session A terminal report / ticket test contract cites `test_iiac_*.py` glob; repo has single module `test_iiac_negative_guards.py` | `T-IGX-016-A.md` test contract | **implementer** | pending |
| **IIAC-GOV-B-F03** | **INFO** | Operator handoff template exists but no filled Session B handoff record (process_enforced assurance only) | `IIAC_SESSION_HANDOFF_RECORD.template.md` | **operator** | n/a |

## Verdict (exact SHA)

**PARTIAL / EXACT BLOCKERS** — governance **implementation quality** (G1–G3 validators, negatives, fail-closed preflights, OD-ENT-02 draft package, activation guard) **satisfies adversarial replay**, but **G0 normative closure inventory is incomplete** @ pinned implementation SHA (**F01**).

**Manifest `verdict` field:** `CONDITIONAL` (single exact blocker before governance PASS).

## Activation eligibility

| Decision | Rationale |
|----------|-----------|
| **`IN_FORCE` in this session** | **Forbidden** (not issued) |
| **Policy activation eligibility** | **NOT ELIGIBLE** until **F01** corrected, matrix regenerated/digested, and certifier path completes (operator activation line + `verify_iiac_policy_activation.py` @ `IN_FORCE`) |
| **EVOL-X / ENT-AUDIT-X IIAC certified** | **Not declared** |

## Handoff

- **Certifier / operator:** after **F01** fix + optional scoped G0 re-audit, publish certification commit with `certification_publication_sha`; pin `session_b_governance_audit_report_digest` in activation JSON.
- **Implementer corrective package:** see [`IIAC_SESSION_B_CORRECTIVE_PACKAGE.md`](./IIAC_SESSION_B_CORRECTIVE_PACKAGE.md).

**Report digest (SHA-256):** see manifest `audit_report_digest` (final value reconciled @ manifest write).
