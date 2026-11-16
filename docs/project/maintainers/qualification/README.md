<!-- Draft added by governance/tickets/T-IGX-042-D.md (EQO-A remediation, documentation only). Not adopted until that ticket carries an Operator approval line. -->
<!-- Sections 7 (ECAP open-question sentence) and 8 were extended by governance/tickets/T-IGX-044-A.md (documentation only; not adopted until that ticket carries an Operator approval line). -->
<!-- Counts in the body were measured with `git ls-tree` at the EQO-A baseline 258bbf6abb6fdbb92ef0185d058527fbb1e47f04 (re-measured 2026-11-16) unless another SHA is named. -->

# Qualification directory - index

Status: **DRAFT - not adopted.** This page is a map. It is **not** a certificate, it carries no status claims, and it does not own anything it points to.

## 1. How to use this directory

`docs/project/maintainers/qualification/` holds the certification, qualification, audit, reconciliation and decision records of the Enterprise program: 808 top-level files (801 `.md`, 7 `.json`) and 13 sub-directories at the EQO-A baseline `258bbf6abb6fdbb92ef0185d058527fbb1e47f04` (2,514 tracked files in total). The Campaign A dossier directory and later PDC/EBH-7 records were added after the baseline: at `4220069c260a5f27bf9f56961155247256541304` there are 14 sub-directories and 2,898 tracked files, and at `7960ae7f6c82da60ca9acdc0ceebf63810002a42` (`origin/development` when this index was prepared, before this change) 14 sub-directories and 2,937 tracked files. Both counts are dated snapshots and drift with every merge. Every file name starts with the program, stage or campaign it belongs to.

Start from the question you have:

| Question | Go to (canonical owner, not this page) |
|---|---|
| What is the status of stage X? | Roadmap `docs/project/maintainers/plans/PLATFORM_ENTERPRISE_COMPLETION_ROADMAP.md` (stage table and ledger) |
| Which FRZ criteria are open? | `PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST.md` |
| What debt, gaps and exceptions are open? | Enterprise Debt & Exceptions Register (`enterprise_debt_register/`) |
| What did a stage prove, at which SHA? | The stage's own certification/qualification record in section 4 |
| Which decisions were taken and by whom? | The operator decision registers and signed decision records in section 5 |
| What did the baseline audit find? | `enterprise_quality_oversight/campaign_a/` (EQO-A dossier, baseline `258bbf6a...`) |

## 2. Authority rule

The index lists *where* a source is and *what kind of record* it is. It never states a stage status, a FRZ status or an owner decision. Where this page and a canonical source disagree, the canonical source wins and this page is corrected. Ownership and status stay in the roadmap, the checklist, the domain architecture documents and the signed decision records.

## 3. Sub-directories (13 at baseline `258bbf6a`; counts are tracked files, recursive)

| Directory | Files | Contents (by name; owner to confirm the one-line role) |
|---|---|---|
| `iiac_evidence/` | 1,024 | IIAC audit-session manifests and evidence per stage |
| `enterprise_quality_oversight/` | not present at baseline (371 at `4220069c`) | EQO-A Campaign A dossier (read-only baseline audit) |
| `evol_x/` | 191 | EVOL-X certification and records |
| `historical_certification_campaign/` | 126 | HIST campaign registers and reports |
| `ent_audit_x/` | 111 | ENT-AUDIT-X records and ARCH-FREEZE handoff |
| `mcp_server_x/` | 75 | MCP-SERVER-X records (incl. `MCP_SERVER_X_INDEX.md`, its own index) |
| `platform_domain_completeness/` | 72 | PDC-X plan, matrices, audits, operator decisions (own index `PDC_INDEX.md`) |
| `scenario_factory_x/` | 46 | SCENARIO-FACTORY-X records |
| `pre_ent_enterprise_certification/` | 35 | PRE-ENT campaign records |
| `ebh_6_p03_independent_replay/` | 14 | EBH-6 Protocol 03 replay evidence |
| `ebh_7_p03_independent_replay/` | 8 | EBH-7 Protocol 03 replay evidence |
| `sef/`, `schemas/`, `examples/` | 2 / 1 / 1 | supporting material |

Existing indexes are linked, not replaced: `EBH_7_EVIDENCE_DOSSIER_INDEX.md`, `mcp_server_x/MCP_SERVER_X_INDEX.md`, `platform_domain_completeness/PDC_INDEX.md`.

## 4. Top-level file families (808 files)

Family = leading name token(s). Counts are a prefix classification made for this draft; 84 files (`OTHER`) have no family rule yet and must be classified by the owning program before the index is published (examples: `AW_7C_*`, `DS-E2E-15J-*`, `MP-5F-*`, `GOV_X*`, `GR1*`, `MEMORY_*`, `LLM_EXTERNAL_*`, `CTRL_X_*`, `DEP2..4_*`, `CERT_CAMPAIGN_*`). The index, once generated, is meant to list every file individually with exactly one family and one program/stage ID; that file list does not exist yet and is not part of this change.

| Family | Files | Join key to the canonical owner |
|---|---|---|
| PROD_Q | 85 | PROD-Q |
| INTEGRAX | 74 | legacy platform certification (ID to confirm) |
| DIAGNOSTICS (`DIAG_*`, `DIAGNOSTIC_*`, `DG_*`) | 62 | Observability / DIAG (ID to confirm) |
| NPSC | 50 | NPSC |
| EBH_7 / EBH_6 / EBH_5 / EBH_3 / EBH_4 | 49 / 32 / 22 / 1 / 1 | EBH-7 / EBH-6 / EBH-5 / EBH-3 / EBH-4 |
| TOOL | 47 | TOOL-LIFE-X / TOOL-RUNTIME |
| DIST_SCALE_X | 46 | DIST-SCALE-X |
| TRACE_X | 37 | TRACE-X |
| IIAC | 31 | IIAC-GOV-1 |
| ROADMAP_REPLAY_X | 29 | ROADMAP-REPLAY-X |
| DUP_X | 22 | DUP-X |
| ENTERPRISE_EXECUTION | 22 | execution scale/resilience (ID to confirm) |
| ARCH_FREEZE | 21 | ARCH-FREEZE |
| PLATFORM_EXECUTION / EXECUTION_* / ENTERPRISE_RUNTIME | 19 | (ID to confirm) |
| EE | 13 | execution engine (ID to confirm) |
| HARDENING | 12 | HARDENING-9 |
| HARNESS | 11 | HARNESS |
| EVOL_X | 10 | EVOL-X |
| RAG, STATE_X | 8, 8 | RAG, STATE-X |
| SELF_HEALING, SKILL_X, W5, COMPAT_X, CONFIG_X | 4, 3, 3, 1, 1 | as named |
| OTHER (unclassified) | 84 | to be assigned by owner |

Record types by name (heuristic, not authoritative): qualification records 177, certification records 91, evidence files 54, closure records 53, audit records 51, matrices 32, reconciliations 32, handoffs 21, operator decision records 17, registers/ledgers/indexes 21, SHA ladders 4, other 238.

## 5. Registers, ledgers and decision registers (28 source registers of the EQO-A A7 scan)

Each is canonical for its own rows. Column "EDER label" is the maintenance label required by the Enterprise Debt & Exceptions Register (`LIVE`, `SNAPSHOT_AS_OF_<sha>`, `SUPERSEDED_BY_EDER`); it is **empty in this draft** because assigning it is the register owner's job.

| Register (A7 id) | Canonical file |
|---|---|
| ARCH-FREEZE B2 debt reconciliation (`AF_B2_DEBT_RECON`) | `ARCH_FREEZE_B2_DEBT_REGISTER_RECONCILIATION.md` |
| ARCH-FREEZE closed-world gap register (`AF_CLOSED_WORLD_GAP`) | `ARCH_FREEZE_CLOSED_WORLD_GAP_REGISTER.md` |
| ARCH-FREEZE operator decisions (`AF_OD`) | `ARCH_FREEZE_OPERATOR_DECISION_REGISTER.md` |
| Annex F gap register (`ANNEX_F_GAP`) | `ent_audit_x/ANNEX_F_GAP_REGISTER.md` |
| Architecture debt register (`ARCH_DEBT`) | `docs/project/technical/guides/ARCHITECTURE_DEBT_REGISTER.md` (outside this directory) |
| Delegated certification decisions (`CERT_DELEGATED_DECISIONS`) | `CERT_CAMPAIGN_DELEGATED_DECISIONS_2026-11-07.md` |
| Diagnostic gap ledger (`DIAG_GAP_LEDGER`) | `DIAGNOSTIC_GAP_LEDGER.md` |
| DUP-X findings / operator decisions | `DUP_X_FINDINGS_REGISTER.md`, `DUP_X_OPERATOR_DECISION_REGISTER.md` |
| EBH-5 / EBH-6 / EBH-7 operator decisions | `EBH_5_OPERATOR_DECISION_REGISTER.md`, `EBH_6_OPERATOR_DECISION_REGISTER.md`, `EBH_7_OPERATOR_DECISION_REGISTER.md` |
| EBH-6 conditional remediation / hardcoding | `EBH_6_CONDITIONAL_REMEDIATION_REGISTER.md`, `EBH_6_CORE_HARDCODING_FINDINGS_REGISTER.md` |
| EBH-7 B6 blockers / gaps / cross-domain risk / skip disposition | `EBH_7_B6_BLOCKER_REGISTER.md`, `EBH_7_GAP_REGISTER.md`, `EBH_7_CROSS_DOMAIN_RISK_MATRIX.md`, `ebh_7_p03_independent_replay/EBH_7_SKIP_DISPOSITION_REGISTER.md` |
| HIST registers (independent audit, Path B deferred, Path B residual, remaining blockers, residual findings, stage SHA registry) | `historical_certification_campaign/HISTORICAL_*` |
| MCP-SERVER-X gap register (`MCP_GAP`) | `mcp_server_x/MCP_SERVER_X_GAP_REGISTER.md` |
| PDC gap register (`PDC_GAP`) | `platform_domain_completeness/PDC_GAP_REGISTER.md` |
| ROADMAP-REPLAY-X operator decisions / hardcoding | `ROADMAP_REPLAY_X_OPERATOR_DECISION_REGISTER.md`, `ROADMAP_REPLAY_X_CORE_HARDCODING_FINDINGS_REGISTER.md` |

Other register-like files at the top level (not part of the A7 scan, listed so nothing is missed): `EBH_6_APP_TYP_DEBT_DISPOSITION_REGISTER.md`, `EBH_6_EVIDENCE_FRESHNESS_LEDGER.md`, `EBH_7_REMEDIATION_EXECUTION_LEDGER.md`, `GOVERNANCE_ARCHITECTURE_REBASE_GAP_LEDGER.md`, `ROADMAP_REPLAY_X_SECTION5_LEDGER_ROW_COUNT_RECONCILIATION.md`, `DUP_X_RESPONSIBILITY_DELTA_*.md`.

## 6. Maintenance

The index is meant to be generated from `git ls-files` plus a hand-maintained family map and checked by the same consistency script as the register (neither script exists at the baseline; both are a separate script change): every tracked file under the directory has exactly one family and one program/stage ID; every register in section 5 has an EDER label; every link resolves. A new file in the directory without a family fails the check. The README itself carries no counts that are not regenerated by the script.

## 7. ECAP modes and their entrypoints (EQO-A remediation draft, as of 2026-11-16; EQOA-A0-006)

The Enterprise Continuous Architecture Assurance Protocol ([`ENTERPRISE_CONTINUOUS_ARCHITECTURE_ASSURANCE_PROTOCOL.md`](ENTERPRISE_CONTINUOUS_ARCHITECTURE_ASSURANCE_PROTOCOL.md)) defines four modes in its section 2. At the baseline there is no single ECAP executable; the modes are exercised through the entrypoints below. This table is an index only and creates no new executable.

| ECAP mode (section) | Entrypoints at baseline |
|---|---|
| Change Delta Audit (§2.A) | ENT-AUDIT-X `CHANGED-SINCE` mode (`scripts/enterprise_audit/audit.sh changed <since-sha>`, `audit.bat`; `intergrax/platform/enterprise_audit/changed_since.py`); `FAST` mode for PR and diagnostic runs |
| Parent Closure Audit (§2.B) | The stage's qualification record plus the IIAC-B6 manifest and certifier gate: `scripts/governance/verify_iiac_audit_manifest.py`, `scripts/governance/verify_iiac_role_separation.py`, `scripts/governance/iiac_activation_gate.py` |
| Full Platform Current-HEAD Audit (§2.C) | ENT-AUDIT-X `FULL` and `FREEZE` modes (`scripts/enterprise_audit/audit.sh full`, `audit.sh freeze`; `scripts/enterprise_audit/run_enterprise_audit.py`); EBH-7 attestation `scripts/qualification/attest_ebh7_f10_execution.py` |
| Post-Freeze Change Assurance (§2.D) | No dedicated entrypoint; the ARCH-FREEZE stage is not closed at the baseline |

Open question: whether a standalone ECAP executable is required is **not decided** (EQO-A decision DP-54: the Operator decides whether one is required and, if so, names an owner and a ticket; as of 2026-11-16 that answer is still open). This index takes no position and does not decide it. One related document is not an executable: [`EBH_7_ECAP_EXECUTABLE_CHECKLIST.md`](EBH_7_ECAP_EXECUTABLE_CHECKLIST.md) maps the ECAP-§2.C steps to test evidence.

## 8. Draft records added by T-IGX-044-A (EQO-A remediation, as of 2026-11-16)

Both are drafts next to the ENT-AUDIT-X records, added by `governance/tickets/T-IGX-044-A.md`; neither is adopted and neither changes a status.

| Record | What it is |
|---|---|
| [`ent_audit_x/ENT_AUDIT_X_IIAC_MANIFEST_DRAFT_POINT_IN_TIME_RECORD.md`](ent_audit_x/ENT_AUDIT_X_IIAC_MANIFEST_DRAFT_POINT_IN_TIME_RECORD.md) | Point-in-time facts about the 3 of 40 IIAC manifests that fail the repository validators (EQOA-A2-006); declares no supersession |
| [`ent_audit_x/FRZ_AUD_10_LIVE_ENFORCEMENT_AMENDMENT_DRAFT.md`](ent_audit_x/FRZ_AUD_10_LIVE_ENFORCEMENT_AMENDMENT_DRAFT.md) | Proposal text for a future FRZ-AUD-10 live-enforcement amendment; does not change the disposition, the checklist or the roadmap |
