# SFX capability inventory (companion)

**Ticket:** T-IGX-037-B · **Base SHA:** `eadb1c2116feb0a2090b234a36fd4182bd6ace8d` · **Data:** [`SFX_CAPABILITY_INVENTORY.json`](SFX_CAPABILITY_INVENTORY.json) (`sfx-capability-inventory.v1`, 37 rows, SFX-CAP-01..37) · **Inventory version:** `sfx-capability-inventory.v1+r1` (JSON field `inventory_version`)

**Revision r1 (review corrections R-W1-02, R-W1-07):** CAP-19 follows the MCP-SERVER-X scoped closure (REQUALIFICATION_REQUIRED → REUSE_WITH_CONFIGURATION); CAP-33 is REUSE_WITH_CONFIGURATION because `DEAD_LETTERED` is absent (SFX-GAP-20); CAP-05 names the worker-scoped UCA symbols (SFX-GAP-21); CAP-08 and CAP-31 are REQUALIFICATION_REQUIRED, in line with the Collaborative Work state in master plan §7.5.

Seed: R0 §5 (rows 01–29) with R1 §5 replacements (08, 12, 26) and additions (30–37). Every code path was checked at the base SHA with `git grep` / `git show` (read-only).

## Field conventions

- `maturity`: the first token is `PRESENT`, `PARTIAL` or `ABSENT`. The PDC requirement ID follows in parentheses when a PDC row exists; otherwise the source of the value is named.
- `qualification`: certification ID @ full SHA, as recorded in `PDC_DOMAIN_REQUIREMENTS_MATRIX.json` or in the roadmap §5 ledger. Every PDC row has `CERTIFICATION_STATUS = NOT_CERTIFIED` at current HEAD, so all cited certifications are **historical**, not current-HEAD proof.
- `evidence_sha`: `code=` is the base SHA for the code path; `cert=` is the closure SHA of the cited certification.
- The PDC matrix's double-encoded "—" (`â€”`, defect RA-F-05) is left as is.

## Counts per REUSE_CLASS

| REUSE_CLASS | Count | Rows |
|---|---|---|
| REUSE_AS_IS | 15 | 01, 02, 03, 05, 06, 07, 09, 12, 14, 16, 17, 20, 24, 32, 34 |
| REUSE_WITH_CONFIGURATION | 11 | 04, 10, 11, 13, 18, 19, 22, 23, 33, 36, 37 |
| EXTEND_CONTRACT | 2 | 21, 25 |
| NEW_SCENARIO_FACTORY_COMPONENT | 2 | 29, 35 |
| REQUALIFICATION_REQUIRED | 4 | 08, 15, 30, 31 |
| DEFERRED_WITH_RATIONALE | 1 | 27 |
| BLOCKED | 2 | 26, 28 |
| **Total** | **37** | |

Mappings from seed wording to one class: CAP-21 "REUSE_AS_IS / EXTEND_CONTRACT" → EXTEND_CONTRACT (the lifecycle mapping is the binding part). CAP-35 "pattern reuse" → NEW_SCENARIO_FACTORY_COMPONENT (new sidecar, no import). CAP-37 "EXTEND usage" → REUSE_WITH_CONFIGURATION (scanner contract unchanged; OD-SFX-14 pending). For CAP-15, CAP-30, and (Collaborative Work, see below) CAP-08 and CAP-31, the requalification flag becomes the primary class; the target class after requalification is in `limitations`. CAP-19 was REQUALIFICATION_REQUIRED in the first publication and is REUSE_WITH_CONFIGURATION at r1, because MCP-SERVER-X is now CLOSED (scoped).

## Path corrections (verified, not missing)

- CAP-02, 13, 14, 19: the seed's `applications/_shared/...` is `intergrax/applications/_shared/...`.
- CAP-19: `compose_application_mcp_server` is defined in `mcp_application_server.py:45`, and `McpExposurePolicy` in `intergrax/contracts/mcp_server_definition.py:91`. `mcp_developer_api.py` only re-exports them.
- CAP-32: `SessionStorage` is defined in `intergrax/memory/contracts/session_storage.py:14`; the Nexus module re-exports it.

## Rows with MISSING_EVIDENCE

| Row | Field | Finding |
|---|---|---|
| CAP-02 | qualification | The seed claims "U1 PASS", but no such record appears in the roadmap or the PDC evidence read. |
| CAP-05 | code_path | The seed names only the FROZEN architecture record (`GOVERNED_CAPABILITY_FULFILLMENT.md:10`). r1 adds the worker-scoped UCA symbols (SFX-GAP-21): `WorkerCapabilityAcquisitionDecisionService` (`intergrax/autonomous_work/capability_acquisition_service.py:100`), `WorkerCapabilityNeed` (`intergrax/contracts/autonomous_work/capability_acquisition.py:181`), `build_worker_capability_fulfillment_coordinator` (`intergrax/autonomous_work/worker_capability_fulfillment_composition.py:24`). They cover worker obstacle recovery only; the SFX-facing entry point stays MISSING_EVIDENCE. The certification UCA-6C-R6-CERT-RERUN-3 @ `a38f70bc…` is recorded. |
| CAP-08 | qualification | MP-4 is FORMALLY CLOSED (`capabilities/plan/MULTIPLAYER_AI.md:213`), but no SHA was found. MP-6G covers decisions only as an activity source. |
| CAP-30 | qualification / evidence_sha | MP-3H "ENTERPRISE CERTIFIED / CLOSED" has no recorded SHA. |
| CAP-31 | qualification | MP-2 (COLLAB-WORK-2F/2G) has no recorded SHA. |
| CAP-33 | code_path | `DEAD_LETTERED` does not exist under `intergrax/` at base. PDC-D7-R013 is ABSENT ("no DLQ anywhere"), and ADR-PDC-07 is a decision record only. At r1 the class is REUSE_WITH_CONFIGURATION (was REUSE_AS_IS): checkpoints and the scheduler ledger are reusable, the dead-letter part is not (SFX-GAP-20). |
| CAP-36 | limitations | Egress enforcement strength depends on the provider. The isolation gate only checks that isolation is present, and fails closed. |
| CAP-03 / CAP-19 | limitations | These carry over the seed gaps SFX-GAP-09 (dual `ToolRegistry`) and SFX-GAP-07 (MCP client compatibility). |

## Rows with REQUALIFICATION_REQUIRED

- **CAP-15** (Nexus `ArtifactStore`, `ArtifactRef`): PDC-D8 has `qualification_suites = []` and `prior_certifications = {}`. D8-R005 and D8-R007 are PARTIAL; D8-R006 and D8-R014 are ABSENT.
- **CAP-30** (Collaborative Work artifacts), **CAP-08** (decision bindings) and **CAP-31** (Shared Work): see the finding below.
- **CAP-19** (MCP exposure) is no longer in this class at r1: MCP-SERVER-X is CLOSED (scoped, P0–P8; §5 ledger and `MCP_SERVER_X_INDEX.md`), and no requalification at current HEAD is recorded. PDC-D6-R017 stays BLOCKED_EXACT_REASON and the scoped limitations stay in force (see `limitations`).

## Collaborative Work (CAP-08 / CAP-30 / CAP-31) qualification finding

- **PDC:** `PDC_CLOSED_WORLD_INVENTORY.json` → `domain_links["PDC-D8"]` has `qualification_suites: []` and `prior_certifications: {}`. No roadmap §5 ledger row covers Collaborative Work.
- **Domain closures** (`maintainers/plans/COLLABORATIVE_WORK.md`): MP-2 CLOSED, with repository suite 3.0.0 QUALIFIED on SQLite and PRODUCTION_QUALIFIED on PostgreSQL. MP-3H ENTERPRISE CERTIFIED. MP-6G CLOSED / CERTIFIED, `qualification_sha 5ff90667569bf23c88a4db98d489966e309f4ab7`. Only MP-6G has a SHA.
- **`tests/qualification`:** no file references `WorkArtifactRepository`, `WorkArtifactVersionRepository`, `ArtifactPublicationRepository`, `ArtifactContentStore`, `AssignmentRepository` or `WorkItemExecutionLinkRepository`.
  - `mp6/mp6g_e2e_contract.py` covers work-item create, replay, transition and cross-tenant/workspace isolation, but covers artifacts only as an activity source (`run_mp6g_artifact_source_coverage`).
  - `multiplayer/mp_final2/host_operability.py` references `WorkItemRepository` and `CollaborativeDecisionBindingRepository`.
- **Result:**
  - **CAP-30 = REQUALIFICATION_REQUIRED.** No SHA-pinned qualification covers the immutable-version, CAS publication or content-store semantics that SFX relies on.
  - **CAP-31 = REQUALIFICATION_REQUIRED (r1; was REUSE_WITH_CONFIGURATION).** MP-6G @ `5ff90667…` covers it at activity scope, and MP-2 covers it without a SHA. A current-HEAD replay of `repository_qualification_suite.py` is still needed before the freeze pin. Target class after requalification: REUSE_WITH_CONFIGURATION.
  - **CAP-08 = REQUALIFICATION_REQUIRED (r1; was REUSE_WITH_CONFIGURATION).** The decision-binding repository is Collaborative Work code; MP-4 closure has no recorded SHA and MP-6G covers decisions only as an activity source. Target class after requalification: REUSE_WITH_CONFIGURATION. The code evidence (three backends verified) does not contradict this: it shows the code exists, not that it is qualified.
  - **Consistency rule:** master plan §7.5 states that the Collaborative Work qualification state is REQUALIFICATION_REQUIRED. Every row whose owner is Collaborative Work and which has no SHA-pinned qualification (CAP-08, CAP-30, CAP-31) therefore carries that class. ADR-SFX-002 D8 and the lifecycle depend on these rows, so SFX claims cannot exceed this state.

## Domains not covered by PDC (D1–D8) and the evidence used instead

| Domain | Rows | Evidence used |
|---|---|---|
| Tool lifecycle | 03 | TOOL-LIFE-X @ `1548640708c850b3fbfca3f2db0e5ecfee4274dd`, CLOSED – INDEPENDENTLY ACCEPTED, **annotated**: AUD-TOOLLIFE CONDITIONAL / chain gate STOP at the same SHA, so the label is not citable as current-HEAD proof (`TOOL_LIFE_X_PARENT_FORMAL_CLOSURE_RECONCILIATION.md`). Indirect PDC link: D7-R009. |
| MCP | 19 | MCP-SERVER-X §5 ledger: CLOSED – INDEPENDENTLY AUDITED (INTERNAL AGENT CERTIFICATION) – PUBLISHED (SCOPED, P0–P8); baseline pin `c228799785db8f35826f47ba0790d6a21d97fbe3`; P7 package `abbe1525450da77da6801b080619a600c20cd497`; P8 audit PASS_SCOPED @ `eadb1c211`; IIAC CERTIFIED_PASS_SCOPED (`919e34b8f`) (`mcp_server_x/MCP_SERVER_X_INDEX.md`, final closure report). Scoped limitations stay in force. PDC-D6-R017 only flags the exposure security. |
| Integrations | 09, 10 | CONFIG-X @ `fac9a43f3b128fde63541d2b56ebd8f4a751e8fb` (FRZ-CFG-01..08 PASS); INT-CONFIG-REAL-X @ `a59744517b92847f55def1db22826d17d89ee155`. The credential broker also has PDC-D6-R009 (TENANT-X). |
| Observability | 14, 34 | TRACE-X @ `a3203d7994bdfd3c1dde2f741aaefcc3896a5cc2` (FRZ-TRC 12/12); CTRL-X @ `318a4d835534ef5c48575f4649551c957dacd8ea`. |
| Governance / execution | 01, 07 | GOV-X2 @ `e2d7a40b291c658677cc6d981183e20957c7cf0e`, which PDC-D6-R003 and -R018 also cite. |
| Configuration | 10, 11 | CONFIG-X (above); HARNESS-W7/HOST-01 @ `5d5442da…` through PDC-D4-R001. |
| RAG | 23 | **Partly covered** by PDC-D2-R004 (EBH-2G @ `ba41ef75…`), D2-R002 (TENANT-X), and D2-R005/R008 (PARTIAL). |
| Skills | 20, 25 | SKILL-X @ `552e379845a9512c275f3ebe520ac51767c97cf8`, **CONDITIONAL** on the QUAL-X regularisation (OD-ROUTE-DS5-2026-11-07). Disable, remove and upgrade are unsupported (PDC-D1-R018 PARTIAL). |
| Tenant isolation | 17 | TENANT-X @ `5ccb72fccd4e3820ebdc9c18008afe6f7d59f558` (FRZ-TEN PASS = 12; `TENANT_X_CROSS_PLATFORM_TENANT_ISOLATION_CERTIFICATION.md`). |
| Capability fulfilment (UCA) | 05 | `GOVERNED_CAPABILITY_FULFILLMENT.md` §3: certified code baseline `a38f70bc878a4e61807ce93cb0bfa600fdaca168`. |
| Scenario tooling / scaffold / Studio | 18, 20–22, 25, 29, 35, 37 | No certification; code verified only (NOT_CERTIFIED). |

## Out of scope

No code or record is changed. The platform-owned seams (SFX-GAP-15: `agent_contract_meta.py:71`, `root.py:85`) and the MCP allowlist widening (SFX-GAP-17: `mcp_application_server.py:83-91`) were re-verified at base and are recorded in the rows only.
