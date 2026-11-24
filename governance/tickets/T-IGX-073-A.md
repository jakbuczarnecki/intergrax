# T-IGX-073-A — AF-02B: PT-FRZ-RR Path B amendment (AD-AF-PTFRZ-001)

## Scope

Documentation-only execution of architecture decision **AD-AF-PTFRZ-001** (AF-02B): formalize PATH B as the substitute substantive basis for PT-FRZ-RR when WP-4 primary artefacts are absent; align MEP/DAG/matrix/register text with AF-02A-G6 evidence at audit SHA `aaabfc6c8d3a0f7dd43ebd43f47d830bdf1c7a20`; disposition findings F-PT-001 and F-PT-003 in-repo without FRZ promotion, node closure, or DS-4 admission.

**Authority:** Architecture decision AD-AF-PTFRZ-001 ACCEPTED (ChatGPT, ARCH-FREEZE); operator re-approval not required for this batch per AF-02B mandate.

## Write set

```text
docs/project/maintainers/qualification/arch_freeze_program/PT_FRZ_RR_PATH_B_AMENDMENT.md
docs/project/maintainers/qualification/arch_freeze_program/MEP-AF2_v2.1.md
docs/project/maintainers/qualification/arch_freeze_program/WP12_DAG.md
docs/project/maintainers/qualification/arch_freeze_program/WP12_TICKETS_DRAFT.md
docs/project/maintainers/qualification/arch_freeze_program/README.md
docs/project/maintainers/qualification/arch_freeze_program/SOURCES.sha256
docs/project/maintainers/qualification/ARCH_FREEZE_OPERATOR_DECISION_REGISTER.md
docs/project/maintainers/qualification/ARCH_FREEZE_FRZ_ACCEPTANCE_MATRIX.md
governance/tickets/T-IGX-073-A.md
```

## Test contract

```text
uv run python scripts/governance/verify_arch_freeze_plan_traceability.py
uv run python scripts/docs/check_docs_domain_pairs.py
git diff --check
```

## Out of scope

`intergrax/`, `agents/`, `applications/`, `scripts/`, `tests/`, `.github/`, roadmap, FRZ checklist (Operator-only), IIAC, `PT_FRZ_RR_CLOSED=YES`, `DS4_READY=YES`, closing 109 OPEN rows, unconditional PASS for G6, recovery search for WP-4.

## Lifecycle state

in_progress