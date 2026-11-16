# PLAN T-IGX-042-B — EQO-A remediation batch DB-F2: decision-dependent owner-document hunks

Status: DRAFT for Operator review. Drafted 2026-11-16 from the WP-9 plan (`.tmp/session/eqo-a-remediation/WP-9/patches/DB-F2/`) after the Operator decisions DP-01..DP-68. Nothing here is approved, applied or audited.

## 1. Purpose

Add scope labels, owner mappings and pointer paragraphs to domain owner documents that the EQO-A Campaign A dossier (PR #589, baseline `258bbf6abb6fdbb92ef0185d058527fbb1e47f04`) found unlabelled, unowned or stale, for the hunks whose blocking open question has since been decided. Documentation only. This ticket certifies nothing, promotes no FRZ criterion, changes no stage status and edits no product path.

## 2. Decision basis (blocker to decision, recommended option A in each case)

| Blocker | Decision | Effect on the text |
|---|---|---|
| W5:OQ-5, W5:OQ-8 | DP-05 | Feature-local certifications are labelled "not IIAC certified; no independent-audit artefact located"; AW-7A/AW-7B independence is recorded as UNVERIFIED; no re-audit is implied |
| W5:OQ-2, W5:OQ-12 | DP-56 | Execution-path domains (Nexus, Orchestration, Reasoning and Cognition, AHI, Elastic Capacity) are assigned to existing owners; no new stage or FRZ family; "assignment proposal, confirmation pending" wording replaced |
| W5:OQ-1 | DP-55 | Decision System is a named sub-domain of GOV-X2, CTRL-X CX-05 and EBH-5/7; DS-E2E-15J is recorded as historical and non-independent |
| W5:OQ-7 | DP-57 | `PROOF_DATA_PACKAGE_DISTRIBUTION` is demoted to proof infrastructure (satellite of QUAL-X / EBH-6); the drafted "open" wording is replaced |
| W5:OQ-9 | DP-57 | Concern-level mapping waits for the Operator's concern list |
| W5:OQ-10 | DP-33 | Experiment registry: reachability check first; scoped limitation if unreachable, fix if reachable; the paragraph states facts and this handling only |
| W9:Q5 | not decided | Hunk H34 is not included |

## 3. Scope (36 hunks, 23 files; H34 excluded)

| Hunks | Files | Findings | Change |
|---|---|---|---|
| H01, H02, H05, H06, H07, H16, H32, H36 | `MEMORY_ARCHITECTURE.md`, `MEMORY.md`, `OBSERVABILITY.md`, `RAG.md`, `UNIFIED_EXECUTION_RUNTIME.md`, `PLATFORM_PLUGINS.md`, `MULTIPLAYER_AI.md`, capabilities `README.md` | A4-029, A8-002, A8-016 | Scope labels naming the covering stage or FRZ family; no ARCH-FREEZE closure |
| H03, H08, H09, H10, H11, H15, H25, H26 | `MEMORY.md`, `NEXUS_EXECUTION_FLOW.md`, `ORCHESTRATION.md`, `REASONING_AND_COGNITION.md`, `SKILLS.md`, `TOOLS.md`, `ADAPTIVE_HARNESS_INTELLIGENCE.md`, `ELASTIC_CAPACITY_AND_SCALING.md` | A8-004, A8-005, A8-007, A8-013, A4-023 | Owner mappings and owner-document tables, by ID |
| H04, H12, H13, H14, H17, H22 | `MEMORY.md`, `TOOLS.md`, `PLATFORM_PLUGINS.md`, `CAPABILITY_CATALOG_AND_DISCOVERY.md` | A8-007, A4-022, A4-023, A8-008 | Pointer paragraph, tool-count correction (200 to 201, gate-asserted), two-plugin-packages note |
| H18, H19, H20, H21, H23, H24 | `AUTONOMOUS_WORK.md` (architecture and plan hub) | A8-008, A8-009, A8-014 | Implemented slice versus target, module groups with owner documents, AW-7A/7B UNVERIFIED |
| H27, H28, H29, H30 | `COLLABORATIVE_WORK.md`, `EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md`, `TIER3_APPLICATION_ENVIRONMENT.md` | A8-014, A8-015 | Owner mapping, recovery and observability ownership pointers, experiment registry facts |
| H31 | `DECISION_SYSTEM.md` | A4-013 | Sub-domain owner mapping |
| H33, H35, H37 | `PROOF_DATA_PACKAGE_DISTRIBUTION.md`, capabilities `README.md` | A8-011, A8-019, A8-023 | Cross-cutting ownership table, EVOL_X and MCP_SERVER index rows, exemption table for architecture files without a plan pair |

Every factual sentence has a `FACT_CHECK.md` entry (`path@sha:line`) in the WP-9 working notes; counts were re-checked against `origin/development` at the head used for the PR.

## 4. Explicitly out of scope

- H34 (`EVOL_X_CHANGE_SAFETY.md` status line, A4-038/A8-018): blocked on W9:Q5, not decided.
- Concern IDs for A8-014 (wait for the Operator's concern list), plan stubs for COMPAT_X / PROOF_DATA / TOKEN cache-prefix (exemption table used instead), TOOLS.md pruned mention lists, `scripts/docs/check_docs_domain_pairs.py` extension, labels in `LANGCHAIN_INDEPENDENCE.md` and `TOKEN_OPTIMIZATION.md`, files owned by batch DB-F1.
- Any file under `intergrax/`, `agents/`, `applications/`, `scripts/`, `tests/`, `.github/`, the roadmap, the FRZ checklist, `governance/` (except this ticket and plan) or `docs/project/maintainers/qualification/`.
- Any certification statement: no sentence added by this batch asserts or upgrades independent verification, PASS or closure; roadmap labels are quoted and attributed, not re-asserted.

## 5. Write set (the only paths the PR may touch)

The 23 files named in section 3 plus this plan and `governance/tickets/T-IGX-042-B.md`.

## 6. Test contract

One process at a time, under the shared pytest lock, nothing broader:
- `uv run python scripts/docs/check_docs_domain_pairs.py`
- `uv run pytest tests/unit/docs/ tests/qualification/pdc_x/p6_repair/test_p6r_feature_docs.py tests/unit/runtime/architecture/test_gov_final_1_documentation_regression_gates.py -p no:cacheprovider`
- Reviewer check: `git diff --stat origin/development...HEAD` lists exactly the 23 files plus the ticket and plan.

## 7. Risks and ordering

- Roadmap labels quoted in notes can drift; the roadmap row decides status.
- The decision record is a session artifact of the EQO-A remediation planning; the documents refer to "the Operator" without citing it until the Operator publishes it.
- Docs under `docs/project/architecture/` and `docs/project/capabilities/` are outside the EBH-7 allowlist; confirm with the Operator, as owner of the EBH-7 pin, that no new re-pin is required before merge.
- A reviewer other than the author must read the owner mappings and the DP-33 / DP-55 / DP-57 wording against the decisions.

## 8. Rollback

Revert the single PR. No data, configuration or code path is affected.

## 9. Closure condition

PR merged after review; the ticket records the merge SHA. The findings covered are marked `DOC_PATCH_APPLIED` in the Enterprise Debt & Exceptions Register when it exists; none is marked closed or verified by this ticket.
