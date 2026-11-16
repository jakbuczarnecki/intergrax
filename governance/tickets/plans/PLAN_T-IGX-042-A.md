# PLAN T-IGX-042-A — EQO-A remediation batch DB-F1: decision-free owner-document corrections

Status: DRAFT for Operator review. Drafted 2026-11-16 by the orchestrating session from the WP-9 plan and applied by an implementation session as a draft pull request. Nothing here is approved or audited.

## 1. Purpose

Correct stale or missing statements in eleven domain owner documents that the EQO-A Campaign A dossier (PR #589, baseline `258bbf6abb6fdbb92ef0185d058527fbb1e47f04`) found inconsistent with the roadmap or the code. Documentation only. This ticket certifies nothing, promotes no FRZ criterion, changes no stage status and edits no product path.

## 2. Scope (18 hunks, 11 files)

| Hunk | File | Findings | Change |
|---|---|---|---|
| H01-H03 | `docs/project/architecture/GOVERNED_EXECUTION.md` | A4-010 | Dated notes under the existing maturity statements pointing to the GOV-X1/GOV-X2 closure rows; the original sentences stay (append-only) |
| H04 | same | A4-033 | Security owner chain (ADR-PDC-06, threat model, D1-D8 satellite) as an owner pointer; the inbound link from `DOCUMENTATION_MAP.md` is NOT part of this batch |
| H05 | `docs/project/architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md` | A4-014 | New section naming the agent contract surfaces (`Agent`, `Tier2Agent`, `RoutableTier2Agent`) and their layering |
| H06-H08, H22 | `docs/project/capabilities/architecture/MCP_SERVER.md`, `docs/project/capabilities/plan/MCP_SERVER.md` | A4-021, A8-018 | Replace the stale status line with a pointer to the roadmap row `MCP-SERVER-X` and its quoted scoped label; rename "Planned logical components" to "Logical components"; dated note on the stale wave table |
| H09 | `docs/project/architecture/CAPABILITY_MARKETPLACE_ENGINE.md` | A4-023 | Name the acquisition/qualification packages and the marketplace tool path (verified counts at baseline); no reachability claim beyond what `git grep` shows |
| H10 | `docs/project/architecture/LLM_ADAPTERS.md` | A4-023 | Name `intergrax/runtime/llm` (3 modules) with its owner stage |
| H11-H13 | `LLM_ADAPTERS.md`, `CONTEXT_ENGINEERING.md`, `MODALITY.md` | A8-007 | One recovery/lifecycle pointer paragraph per domain document, naming the owner by ID only |
| H14 | `docs/project/architecture/DIAGNOSTICS.md` | A7-002 | Mark the DG-003/DG-005 ledger statuses as point-in-time |
| H15-H16 | `docs/project/architecture/ENTERPRISE_RELIABILITY_LAYER.md` | A8-010 | Restate maturity as implemented slice vs target; replace the "no plan pair" statement with a link to the implementation plan |
| H17 | `docs/project/capabilities/architecture/COMPAT_X_CONTRACT_SCHEMA_EVOLUTION.md` | A8-011 | Record that no plan pair exists and point to the roadmap rows; the draft's cross-reference to a `capabilities/README.md` exemption table is dropped because that table is created by batch DB-F2, not by this batch |

Source of the drafts: `WP-9/patches/DB-F1/PATCH.diff` restricted to the 18 hunks above (61 insertions, 4 deletions before the two adjustments in section 5). Every factual statement has a `FACT_CHECK.md` entry with `path@sha:line`.

## 3. Explicitly out of scope

- H18, H19 (`TOKEN_OPTIMIZATION_CACHE_PREFIX_STABILIZATION.md`, `ADR_PDC_05_EVALUATION_CONTRACTS.md`, finding A4-038): blocked on Operator question W9:Q5 (coordination with PDC-owned records).
- H20, H21 (`SCENARIO_FACTORY.md` link, new `SFX_SCENARIO_MECHANISM_MAP.md`, finding A4-047): the map needs one Operator fact (owner of the scenario-readiness contracts, still a placeholder). H20 only links to the map from H21, so both stay out together (WP-9 counted 19 decision-free hunks; the dependency of H20 on H21 makes the self-contained set 18).
- Any file under `intergrax/`, `agents/`, `applications/`, `scripts/`, `tests/`, `.github/`, the roadmap, the FRZ checklist, `governance/` or `docs/project/maintainers/qualification/`.
- Any certification statement: no sentence added by this batch asserts or upgrades independent verification, PASS or closure; roadmap labels are quoted and attributed, not re-asserted.

## 4. Write set (the only paths the PR may touch)

```text
docs/project/architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md
docs/project/architecture/CAPABILITY_MARKETPLACE_ENGINE.md
docs/project/architecture/CONTEXT_ENGINEERING.md
docs/project/architecture/DIAGNOSTICS.md
docs/project/architecture/ENTERPRISE_RELIABILITY_LAYER.md
docs/project/architecture/GOVERNED_EXECUTION.md
docs/project/architecture/LLM_ADAPTERS.md
docs/project/architecture/MODALITY.md
docs/project/capabilities/architecture/COMPAT_X_CONTRACT_SCHEMA_EVOLUTION.md
docs/project/capabilities/architecture/MCP_SERVER.md
docs/project/capabilities/plan/MCP_SERVER.md
```

Plus the ticket and this plan under `governance/tickets/` and `governance/tickets/plans/`. Nothing else.

## 5. Method

1. Create a worktree outside the repository (`<parent>/_wt/intergrax/db-f1`) from `origin/development`; new branch `docs/eqo-a-db-f1-owner-docs` (branch creation needs the Operator's approval, see `agent-env/rules/intergrax-git-branch-safety.md`).
2. Apply the patch (`git apply --check` first, then apply). It was drafted against `origin/development` `665760ceb06aee062cf254f485e895d7e928ec15` and re-checked at `7960ae7f6c82da60ca9acdc0ceebf63810002a42`: applies cleanly with one whitespace warning (two trailing spaces at the end of the added `**Status:**` line in `capabilities/architecture/MCP_SERVER.md`). Two adjustments after the re-check: (a) those two trailing spaces are removed (the file stays CRLF; the Status line is no longer a markdown hard break and renders joined to the following `**Pair:**` line); (b) the H17 cross-reference to the not-yet-existing README table is dropped (see the table above). The quoted GOVERNANCE-FINAL verdict in H02 uses the em dash exactly as the record does.
3. Each added sentence was re-read against the head used for the PR (`7960ae7f6`): roadmap rows `GOV-X1`, `GOV-X2`, `MCP-SERVER-X`, `COMPAT-X` and ledger rows `GOV-X1-GR11`, `GOV-X1-GR13` are unchanged from the baseline; the file counts and line counts of the cited code directories match the baseline; the product-path changes between baseline and head touch only `intergrax/rag/graph/`; stale statements are fixed or dropped, never kept.
4. Commit without any AI-tool attribution. The `Ref:` trailer is included although the paths are docs-only.

## 6. Test contract (none run at drafting time)

One verification step, one process at a time (`agent-env/rules/intergrax-test-process-budget.md`):
- `uv run pytest tests/unit/runtime/architecture/test_gov_final_1_documentation_regression_gates.py tests/unit/runtime/architecture/test_gov_final_3_*.py tests/unit/runtime/architecture/test_gov_final_4_*.py tests/unit/docs/ -p no:cacheprovider` (these parse `GOVERNED_EXECUTION.md` and the docs tree).
- `uv run python scripts/docs/check_docs_domain_pairs.py` (the plan-pair statements for the reliability layer and COMPAT-X).
- B6-F10 / `RF-GOV-01` binding: `GOVERNED_EXECUTION.md` is the owner document bound to `RF-GOV-01` in `tests/qualification/ebh_7/_ebh7_f10_canonical_inputs.py`. The implementation session was not authorised to run qualification suites, so this item is a static read only (result in the PR body); the `DOCUMENTATION_ONLY` class regression named in that file is `tests/qualification/ebh_7/test_ebh_7_b6_f10_f11_corrective_gates.py::test_b6_f10_matrix_markdown_matches_generated`. Running it, if wanted, is a separate Operator decision.
- Reviewer check: `git diff --stat origin/development...HEAD` lists exactly the 11 files plus the ticket and plan.

## 7. Risks and ordering

- **EBH-7 pin interplay.** At `origin/development` `7960ae7f6` the last commit that re-pinned `TEST_EXECUTION_SHA` is `ee8b37925` (to `d39a520dd`, 2026-11-15); later commits under `tests/qualification/ebh_7` (the last is `e5d428da6`, 2026-11-16) edit tests, not the pin, and the PDC-X closure PRs #603 and #604 are merged. No pin is in flight in the git history. Static finding: the owner document is only path-resolved by the F10 inputs (`_resolves(owner_doc, inputs)`), its content is not parsed; paths changed between `TEST_EXECUTION_SHA` and `HEAD` are compared with an allowlist (`scripts/qualification/verify_ebh7_p03_immutable_evidence.py`) that does not list `docs/project/architecture/` or `docs/project/capabilities/`, so `identity_violations` would name these files, as it already names the product and ticket paths that landed after the pin; the C1-C4 closure gate tests replace `identity_violations` with an empty tuple. Whether a replay at a head containing this PR is affected was not run and is not established. Merge only after the Operator or the PDC-X owner confirms this reading.
- **Content risk.** Statements about roadmap labels can drift. Mitigation: quoted, attributed, and tied to the baseline SHA; the roadmap row decides status.
- **Overlap.** Another batch (DB-G, DOCUMENTATION_MAP and the register) owns the inbound link for A4-033; no file is shared with other batches (checked by WP-9 file ownership).
- **Review.** Edits touch statements about GOV, MCP and COMPAT-X closure. A reviewer other than the author must read H01-H03 and H06-H08 against the roadmap rows.

## 8. Rollback

Revert the single PR. No data, configuration or code path is affected.

## 9. Closure condition

PR merged after review; the ticket records the merge SHA; findings A4-010, A4-014, A4-021, A8-007, A8-010, A8-011, A7-002 are marked `DOC_PATCH_APPLIED` in the Enterprise Debt & Exceptions Register when it exists, and A4-023, A4-033 as partially covered (remaining parts in other batches). No finding is marked closed or verified by this ticket.
