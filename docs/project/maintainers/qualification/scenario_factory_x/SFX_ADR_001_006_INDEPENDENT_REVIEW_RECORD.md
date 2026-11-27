# SFX Pre-Freeze Independent Review Record (`ADR-SFX-001`..`ADR-SFX-006` & R1 Schemas)

| Field | Value |
|---|---|
| **Review Session** | `W2-SFX-REV` (`igx-reviewer` subagent `c1d0469e-608e-4664-8695-96bf68875f7a`, read-only, no shell) |
| **Review Date** | 2026-11-27 |
| **Executing Ticket** | [`governance/tickets/T-IGX-090-D.md`](../../../../../governance/tickets/T-IGX-090-D.md) (`OP-EC-01` / `OD-W2-SFX-PREP=A`) |
| **Verdict** | **`ACCEPT_WITH_MINOR_DOC_PATCH`** |
| **Findings Summary** | 0 BLOCKER · 0 CRITICAL · 0 HIGH · 7 MEDIUM (remediated pre-acceptance) · 5 LOW (remediated pre-acceptance) |
| **ADR Posture** | `ADR-SFX-001`..`ADR-SFX-006` remain **`Proposed`** until Operator signs `OD-SFX-06` and `GATE-SFX-ENTRY` |
| **Product Code Delta** | `0` (docs and proposed JSON design schemas only; `intergrax/`, `agents/`, `applications/`, `scripts/`, `tests/` untouched) |

---

## 1. Non-Claims & Governance Boundary

This record preserves all mandatory Intergrax governance boundaries:

- This review declares **NO `G6 PASS`**, **NO `IIAC PASS`**, **NO `FRZ PASS`**, **NO `CLOSED`**, and **NO `SFX_PRE_IMPLEMENTATION_READY`**.
- `ADR-SFX-001` through `ADR-SFX-006` remain **`Proposed`** (`not Accepted`); formal acceptance requires Operator decision **`OD-SFX-06`** and satisfaction of **`GATE-SFX-ENTRY`** after **`ARCH-FREEZE = CLOSED`**.
- `IMPLEMENTATION_AUTHORIZED` remains **`NO`**.

---

## 2. Per-ADR Independent Review Findings & Pre-Acceptance Remediations Applied (`T-IGX-090-D`)

All 7 MEDIUM and 5 LOW findings identified by `W2-SFX-REV` across `ADR-SFX-001..006` have been applied verbatim under `T-IGX-090-D` while preserving `Status: Proposed` on all 6 ADRs:

| ADR | Review Verdict | Remediated Citations | Remediations Applied Pre-Acceptance |
|---|---|---|---|
| **[`ADR-SFX-001`](../../../technical/adr/entries/2026-11-12/ADR-SFX-001.md)** | `ACCEPT_WITH_MINOR_DOC_PATCH` | `:48`, `:46-65`, `:129-131` | 1. Updated scenario execution chain to `execute_scenario_task` (`scenario_runtime_baseline.py:569` at `eadb1c2116fe`; `:592` at `646e63953`) `-> HostTaskExecutionPort` (`intergrax/runtime/execution/host_task.py`, `:616`) `-> NexusLoop` (`:215`) `-> PolicyEngine` (`:46`) `-> ToolRuntime` (`:134`).<br>2. Added `Tenant isolation` row to the no-duplicate-mechanism table (`verify_tenant_security` / `TenantIsolationCheck`, `intergrax/runtime/architecture/tenant_security.py:35`, `SFX-CAP-17`; `RequestIdentity` spine; `PLATFORM-INV-003`, `FRZ-TEN`).<br>3. Added strict tenant child-scope containment (`tenant_child_scope <= tenant_parent_scope`, fail-closed on mismatch, `SFX-CAP-17`) under `Compliance`. |
| **[`ADR-SFX-002`](../../../technical/adr/entries/2026-11-12/ADR-SFX-002.md)** | `ACCEPT_WITH_MINOR_DOC_PATCH` | `:67`, `:115`, `:124-126`; 13 R1 schemas | 1. Clarified that the common envelope applies to CW `WorkArtifactVersion` records (`SFX-CON-01..12, 14..17`), while `SFX-CON-13` (`ScenarioApprovalRef`) is a binding view carrying `tenant_id`, `workspace_id`, `work_item_id`, `binding_id`, `approver`, `canonical_digest`, and `created_at`.<br>2. Added explicit cross-record containment invariant `tenant_child_scope <= tenant_parent_scope` (fail-closed) at the end of D5 and D6.<br>3. Noted at D6 (`:124`) that `SANDBOX_CERTIFIED->ADMISSION_DOSSIER_READY` (`DOSSIER_EXPORT`) follows pending `OD-SFX-18`.<br>4. Closed `WT-12` across all 13 companion R1 schemas (see §3 below). |
| **[`ADR-SFX-003`](../../../technical/adr/entries/2026-11-12/ADR-SFX-003.md)** | `ACCEPT_WITH_MINOR_DOC_PATCH` | `:64`, `:113`, `:121` | 1. Mandated non-empty verified `subject` and `tenant_id`, `tenant_required=True` (HTTP 401/403 on missing/empty, never synthesizing `"default"`), and `reject_identity_assertion_conflicts` in Decision 3.<br>2. Noted `ME-3` traced in `SFX_GAP_ANALYSIS.md §2.3` (Tier-3 host branch of DX-MAINT-04 applies directly; Phase K is not needed). |
| **[`ADR-SFX-004`](../../../technical/adr/entries/2026-11-12/ADR-SFX-004.md)** | `ACCEPT_WITH_MINOR_DOC_PATCH` | `:95`, `:155` | 1. Mandated `fails_closed=True` and `identity_mode=McpIdentityMode.HARNESS_BOUND` on `McpServerCompositionDeps` (never `LOCAL_DEV_EXPLICIT`), `tenant_required=True` on `HarnessAuthState` (no local-dev bypass, no `"default"`/`"mcp"` synthesis), `reject_identity_assertion_conflicts`, and cross-referenced `SFX-GAP-25` / pending `OD-SFX-17` (API-key capped at L0/L1; L2/approvals/HO-2 require human-attributable IdP bearer principal).<br>2. Cross-referenced `SFX-GAP-25` / `OD-SFX-17` in `R-004-03`. |
| **[`ADR-SFX-005`](../../../technical/adr/entries/2026-11-12/ADR-SFX-005.md)** | `ACCEPT_WITH_MINOR_DOC_PATCH` | `:43`, `:72` | 1. Corrected `WorkArtifactVersionRepository.publish_version` to `ArtifactPublicationRepository.publish_version` (`intergrax/collaborative_work/repository.py:1338`, `:1351`).<br>2. Mandated that any derived catalog read model strictly preserves `tenant_id` + `workspace_id` isolation. |
| **[`ADR-SFX-006`](../../../technical/adr/entries/2026-11-12/ADR-SFX-006.md)** | `ACCEPT_WITH_MINOR_DOC_PATCH` | `:124`, `:147`, `:171`, `:198-205` | 1. Replaced `GAP-23` row with exact citation `intergrax/applications/_shared/harness_meaningful_side_effect_authorization_wiring.py:137-157` (`explicit` at `:151`, `strict` at `:155`, `None` on non-STRICT `:156-157`).<br>2. Added explicit MSE authorization port overrides (`harness_meaningful_side_effect_authorization_wiring.py:151`) to the `SFX-P9-02` forbidden list.<br>3. Marked `GAP-23` resolved pre-acceptance under `Implementation notes`.<br>4. Added `tenant_child_scope <= tenant_parent_scope` containment under `Compliance`.<br>5. Cross-referenced pending `OD-SFX-17` / `OD-SFX-18` in §8 and `Compliance`. |

---

## 3. `WT-12` Closure Across Companion R1 Schemas (30 Arrays + 4 Tightenings)

All 13 companion JSON Schema drafts under `docs/project/maintainers/qualification/scenario_factory_x/design/schemas/` were updated with explicit CF-7 array ordering / set-semantics `"description"` annotations (30 previously undeclared arrays + 4 tightened sort-only arrays) and validated as syntactically valid JSON / Draft 2020-12 schemas:

| # | Schema File | CF-7 Arrays Closed / Tightened |
|---|---|---|
| 1 | `sfx.business_problem_statement.v1.schema.json` | `stakeholders` (set, sorted by `role`), `constraints` (ordered), `success_criteria` (set, sorted by `criterion_id`) |
| 2 | `sfx.discovery_record.v1.schema.json` | `open_questions` (ordered as recorded) |
| 3 | `sfx.business_scenario_specification.v1.schema.json` | `actors` (`actor_id`), `inputs` (`input_id`), `outputs` (`output_id`), `business_rules` (`rule_id`), `decision_points` (`decision_id`), `integrations_required` (`integration_id`), `acceptance_criteria` (`criterion_id`) — all set semantics, rejects duplicates |
| 4 | `sfx.scenario_blueprint.v1.schema.json` | `agents` (`agent_id`); tightened `tool_allowlist` (`Set semantics; sorted by tool_id, rejects duplicates.`) |
| 5 | `sfx.integration_setup_plan.v1.schema.json` | `integrations` (`integration_id`), `integrations[].secret_refs` (`name`), `integrations[].network_egress.allowlist` (lexicographical set) |
| 6 | `sfx.scenario_test_plan.v1.schema.json` | `test_cases` (`test_id`), `test_cases[].acceptance_criterion_ids` (lexicographical set), `eval_suites` (`suite_id`) |
| 7 | `sfx.scenario_run_report.v1.schema.json` | `test_results` (`test_id`), `evidence_refs` (lexicographical set) |
| 8 | `sfx.scenario_value_report.v1.schema.json` | `run_report_refs` (`(artifact_id, version)`), `assumptions` (`assumption_id`) |
| 9 | `sfx.scenario_catalog_entry.v1.schema.json` | `tags` (lexicographical set) |
| 10 | `sfx.sfx_session_state.v1.schema.json` | `active_artifact_refs` (`(schema_id, artifact_id)`); tightened `pending_question_ids` (`Set semantics; sorted ascending, rejects duplicates.`) |
| 11 | `sfx.scenario_lifecycle_record.v1.schema.json` | `evidence_refs` (lexicographical set) |
| 12 | `sfx.scenario_qualification_record.v1.schema.json` | `limitations` (lexicographical set); tightened `checks` (`Set semantics; sorted by check_id, rejects duplicates.`) |
| 13 | `sfx.scenario_admission_dossier.v1.schema.json` | `qualification_record_refs` (`(artifact_id, version)`), `security_gate_results` (`gate_id`), `required_integrations_for_prod_q` (`integration_id`), `known_limitations` (lexicographical set); tightened `canonical_artifacts` (`Set semantics; sorted by (schema_id, artifact_id), rejects duplicates.`) |

---

## 4. Pre-Freeze Hygiene Note — `GEN_SET.json` Scratch Relocation (`PF-4`)

- **Current custody risk:** The sealed generalization problem set (`GEN-01..06`, seal `SFX-GEN-SEAL-01`, SHA-256 `8b6ff8eb...3e6e0d` recorded in [`SFX_GEN_SET_SEAL.md`](SFX_GEN_SET_SEAL.md)) currently resides unencrypted in a shared local scratch folder (`D:\Projekty\_wt\_scratch\sfx-gen-set\GEN_SET.{json,md}`).
- **Integrity vs. confidentiality:** The published digest proves tamper-evidence/integrity only; it does **not** enforce confidentiality against coding agents operating under `D:\Projekty\_wt\`.
- **Required operator action before Wave 6 W3 / `SFX-P12-04`:** Operator MUST relocate `GEN_SET.json` and `GEN_SET.md` to an access-restricted location outside all agent workspaces, re-verify the SHA-256 digest matches `SFX-GEN-SEAL-01`, and append the custody relocation note to [`SFX_GEN_SET_SEAL.md`](SFX_GEN_SET_SEAL.md). Agents must never read or inspect `GEN_SET.json` prior to the independent `SFX-P12-04` reveal.
