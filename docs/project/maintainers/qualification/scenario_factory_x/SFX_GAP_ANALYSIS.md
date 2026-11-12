# SFX gap analysis — consolidation (SFX-P2, T-IGX-037-C)

| Field | Value |
|---|---|
| **Ticket** | T-IGX-037-C (`plan_approved`; plan `governance/tickets/plans/PLAN_T-IGX-037-C.md`, deliverable 3) |
| **Capability** | SCENARIO-FACTORY-X (SFX) |
| **Code evidence pin** | `eadb1c2116feb0a2090b234a36fd4182bd6ace8d`. `git diff --stat eadb1c2116fe origin/development -- intergrax agents applications scripts` is empty at `origin/development` = `c11184164`, so every code citation below also holds at that commit [VERIFIED]. |
| **Inputs** | SFX master plan R1 §3.4, §6, §10, §16 (session draft `.tmp/session/sfx/SFX_MASTER_PLAN_R1.md`; published copy `governance/tickets/plans/SFX-MASTER-PLAN.md` via T-IGX-037-A) · W0 register `docs/project/maintainers/qualification/scenario_factory_x/SFX_GAP_REGISTER.md` (worktree `sfx-p0`, unpublished) · `SFX_CAPABILITY_INVENTORY.json`/`.md` and `SFX_IDE_TOOLING_FACTS.md` (T-IGX-037-B) · ADR-SFX-001..006 (all **Proposed**) · `SFX_UX_DESIGN.md` UX-Q-01..08 and `SFX_GEN_SET_SEAL.md` (T-IGX-037-D) |
| **Status vocabulary** | **VERIFIED** — confirmed from a cited path:line or record · **PROPOSED** — a recommendation or design not yet accepted · **MISSING_EVIDENCE** — not yet established · **BLOCKED** — cannot progress until a named external condition holds |
| **Disposition vocabulary** | `RESOLVED` (no further SFX action) · `SCHEDULED` (owned by a named SFX task) · `AVOIDED` (SFX design does not use the deficient mechanism) · `OD_REQUIRED` (operator decision) · `EXTERNAL_DEPENDENCY` (another stage must deliver) · `PLATFORM_HANDOFF` (platform-owned; SFX records and compensates only) |
| **Authority** | This file is documentation. It accepts no ADR, signs no OD, changes no FRZ status and authorizes no SFX-P5+ work (OD-SFX-04, OD-SFX-15). Once published it supersedes the "authoritative until consolidated" clause of `SFX_GAP_REGISTER.md` for dispositions only. |

---

## 1. Summary

**Gaps (27 rows: SFX-GAP-01..17 from R1, SFX-GAP-18..27 new).**

| Gap status | Count | Gaps |
|---|---|---|
| VERIFIED | 23 | 01–06, 08–10, 12–20, 22–26 |
| VERIFIED, with a MISSING_EVIDENCE residual (own row in §2) | 2 | 07 (live interop R-01..R-04), 27 (Cursor de-duplication behaviour) |
| MISSING_EVIDENCE | 2 | 11 (competitive matrix SFX-P1-01 not found), 21 (no verified SFX-facing UCA entry point) |
| **Total distinct gaps** | **27** | |

| Disposition | Count | Gaps |
|---|---|---|
| RESOLVED | 2 | 09, 23 |
| SCHEDULED | 14 | 01, 02, 03, 04, 06, 07, 10, 11, 12, 14, 17, 18, 21, 22 |
| AVOIDED | 1 | 08 |
| OD_REQUIRED | 4 | 13, 16, 24, 25 |
| EXTERNAL_DEPENDENCY | 2 | 05, 20 |
| PLATFORM_HANDOFF | 4 | 15, 19, 26, 27 |

| Blocks | Count | Gaps |
|---|---|---|
| W3 entry | 3 | 10 (via OD-SFX-06), 13 (OD-SFX-14), 24 (OD-SFX-05/09) |
| specific unit | 18 | 01, 02, 03, 04, 05, 06, 07, 11, 12, 14, 17, 18, 19, 20, 21, 22, 25, 27 |
| production admission only | 2 | 15, 16 |
| nothing | 4 | 08, 09, 23, 26 |

**Conflicts:** 9 tracked. 1 RESOLVED (C-OD-01, by OD-SFX-16). 1 PROPOSED resolution awaiting OD-SFX-06 (C-ECAP-01). 7 OPEN (C-SEF-01..04, C-RDY-01, C-AF-01, C-IIAC-01).

**Operator decisions:** 10 R1 decisions open (OD-SFX-05..14). 4 new decisions are PROPOSED (OD-SFX-17..20). UX-Q-04/05 and OQ-002-1..6 route to the OD-SFX-06 ADR review. UX-Q-07/08 map to OD-SFX-07/08.

**GATE-SFX-ENTRY:** **BLOCKED.** ARCH-FREEZE is OPEN, no ADR is Accepted, OD-SFX-05/09/14 are unsigned, `FREEZE_CERTIFICATION_SHA` is unset, and the GEN seal is not yet published (§5).

---

## 2. Gap table

Owner `SFX` means the SFX campaign. Task IDs refer to master plan R1 §10.

### 2.1 SFX-GAP-01..17 (R1 register, final dispositions)

| Gap | Description | Evidence (path:line or doc) | Status | Disposition | Task(s) | Owner | Blocks |
|---|---|---|---|---|---|---|---|
| SFX-GAP-01 | No web toolchain, browser auth or SPA serving | No workflow installs Node (`.github/workflows/unit-tests.yml`, e.g. `:97`); no `StaticFiles` or SPA fallback under `intergrax/` or `applications/`; JWT provider unwired (ADR-SFX-003 "Verified facts") | VERIFIED | SCHEDULED. ADR-SFX-003 S-1..S-7 [PROPOSED]; toolchain policy is OD-SFX-05 | SFX-P5-06, SFX-P10-01, SFX-P10-04..07 | SFX | specific unit (SFX-P10-04 needs OD-SFX-05) |
| SFX-GAP-02 | No scenario catalog | `ApplicationManifest` has `extra="forbid"` and no metadata slot (inventory SFX-CAP-12); pattern `AgentCatalogEntry` (SFX-CAP-35) | VERIFIED | SCHEDULED. Tier-2 sidecar `sfx.scenario_catalog_entry.v1`, manifest unchanged (ADR-SFX-005) [PROPOSED] | SFX-P11-03, SFX-P13-02 | SFX | specific unit (SFX-P11-03) |
| SFX-GAP-03 | Discovery agent absent | No SFX code exists (inventory CAP-29 `NEW_SCENARIO_FACTORY_COMPONENT`) | VERIFIED | SCHEDULED | SFX-P6-01 | SFX | specific unit (SFX-P6-01) |
| SFX-GAP-04 | ROI component absent | PDC-D5-R012 cost-to-success comparison PARTIAL (inventory, `SFX_CAPABILITY_INVENTORY.json:194`) | VERIFIED | SCHEDULED | SFX-P11-02 | SFX | specific unit (SFX-P11-02) |
| SFX-GAP-05 | SEF contracts PROPOSED_ONLY | SFX-CAP-26 "No code (verified absent)"; `ADR-SEF-001` Status **Proposed** at `origin/development` `c11184164` (line 5) | VERIFIED | EXTERNAL_DEPENDENCY. BLOCKED until GATE-SEF-CONSUME (R1 §3.3); `DETERMINISTIC_SCAFFOLD` is unaffected | SFX-P8-03, SFX-P12-01..03 | SEF-X track (+ OD-SFX-10) | specific unit (SFX-P8-03, W7 agentic runs, GATE-SFX-FULL) |
| SFX-GAP-06 | `sync_agent_env.py` generates no agent definitions | `sync_agent_env.py:9-12`, `:148-178` (FACTS summary table); v0 personas work by plain file read (FACTS §5.3) | VERIFIED | SCHEDULED in W3. OD-SFX-16 keeps `sync_agent_env.py` untouched before the freeze; OD-SFX-12 (optional) could pull it earlier | SFX-P4-B01, SFX-P4-B02, SFX-P10-08 | SFX | specific unit (SFX-P4-B01; v0 SFX-P4-A01/A02 not blocked) |
| SFX-GAP-07 | MCP client compatibility | FACTS §5.5 verdict RESOLVED: streamable HTTP and header auth in both clients [CC-MCP], [CUR-MCP]; matches `/mcp` (`fastapi_mcp.py:61-65`) | VERIFIED (doc level) | SCHEDULED. Live residuals R-01..R-04 go to SFX-P10 | SFX-P10-02, SFX-P10-08 | SFX | specific unit (SFX-P10-02 acceptance) |
| SFX-GAP-07 (residual) | Protocol revision negotiation (R-01, R-02), client behaviour on harness 401 without `WWW-Authenticate` (R-03, `harness_auth.py:349-352`), live interop to `/mcp` (R-04) | FACTS §5.4–§5.5 | MISSING_EVIDENCE | SCHEDULED (live proof under `.tmp/proof/sfx/SFX-P10-02/`) | SFX-P10-02 | SFX | specific unit (SFX-P10-02) |
| SFX-GAP-08 | `ApprovalStorePort` is in-memory with no tenant | `intergrax/contracts/agent_runtime_governance.py:366`; `intergrax/runtime/agent_governance/approval_boundary.py:18` (inventory CAP-08 limitations) | VERIFIED | AVOIDED. Approvals use `CollaborativeDecisionBindingRepository` (`collaborative_work/repository.py:1414`); ADR-SFX-002 D8 [PROPOSED] | SFX-P5-05 | SFX | nothing |
| SFX-GAP-09 | Dual `ToolRegistry` ownership | ADR-SFX-001 trace: ToolRuntime resolves only `ToolRegistryRead` (`tool_invoker_protocol.py:21`; `catalog_dispatch.py:57-62`); legacy `tools_base.py:110` deprecated (`:117`) with no production importer, guarded by `tests/qualification/dup_x/test_dup_x_tools_base_production_import_gate.py:38` | VERIFIED (was MISSING_EVIDENCE) | RESOLVED. No platform action. Carry-over [PROPOSED]: SFX-P9-02 forbids `intergrax.tools.tools_base` imports in generated packages | SFX-P9-02 (carry-over only) | SFX | nothing |
| SFX-GAP-10 | Business-scenario workspace location | ADR-SFX-002 D4: scaffold output under `applications/`; slug rule `scripts/proof/create_scenario_proof.py:19` | VERIFIED | SCHEDULED through ADR-SFX-002 D4 [PROPOSED]. OQ-002-1 (path form) is open | ADR acceptance (OD-SFX-06); SFX-P8-01 | SFX / operator | W3 entry (ADR-SFX-002 acceptance) |
| SFX-GAP-11 | Research facts | IDE/MCP facts complete (`SFX_IDE_TOOLING_FACTS.md`) [VERIFIED]; inventory re-seeded, 37 rows (`SFX_CAPABILITY_INVENTORY.md`) [VERIFIED]; `SFX_COMPETITIVE_EVIDENCE_MATRIX.md` (SFX-P1-01) is not present in the `sfx-w1-research` worktree at the time of writing | MISSING_EVIDENCE (P1-01 only) | SCHEDULED (T-IGX-037-B closeout) | SFX-P1-01 | SFX research lanes | specific unit (T-IGX-037-B closure; PRE_IMPLEMENTATION_READY) |
| SFX-GAP-12 | No SFX qualification suite | No `tests/qualification/scenario_factory_x/` exists (SFX code absent) | VERIFIED | SCHEDULED | SFX-P9-01..05, SFX-P10-03, SFX-P13-01..03 | SFX | specific unit (W5 and W8 acceptance) |
| SFX-GAP-13 | No PR-level SAST or dependency audit; ruff has no `S` rules; no pip-audit or npm audit | `pyproject.toml:390` `select = ["E", "F", "I", "UP"]`; scanner references only in `.github/workflows/harness-release.yml` | VERIFIED | OD_REQUIRED (OD-SFX-14), then SCHEDULED | SFX-P9-03, SFX-P10-04 | operator → SFX | W3 entry (OD-SFX-14 is an entry condition) |
| SFX-GAP-14 | No conformance rule forbids generated code from mutating environment profile, policy mode, governance or certification records, contract lifecycle or exposure policy | `scripts/proof/scenario_architecture_conformance.py:256-340` covers only `AgentRegistry` construction and mutation | VERIFIED | SCHEDULED. Rule list extended [PROPOSED] with the `tools_base` import (GAP-09) and the MSE `explicit` override (GAP-23) | SFX-P9-02 | SFX | specific unit (SFX-P9-02 must pass before any L2 run of generated code) |
| SFX-GAP-15 | Platform self-promotion seams S1–S6 | ADR-SFX-006 §7: `agent_contract_meta.py:71`; `environment_profile/root.py:85`; `scenario_runtime_profiles.py:114-159`; `agent_promotion.py:36`; `agent_certification_wiring.py:35-78`; `scripts/proof/scenario_lifecycle.py:29` | VERIFIED | PLATFORM_HANDOFF (Agent Runtime Governance / PDC-X D1, D6). SFX compensates in SFX-P9-02 and holds no L3 credentials | SFX-P9-02; handoff record per OD-SFX-20 | platform owner | production admission only |
| SFX-GAP-16 | No owner for per-scenario production admission | ADR-SFX-006 §6: existing gates cover agents, plugins and integrations only | VERIFIED | OD_REQUIRED (OD-SFX-13) | SFX-P11-04 (dossier only) | operator | production admission only |
| SFX-GAP-17 | MCP composition widens `allowed_binding_ids` | `intergrax/applications/_shared/mcp_application_server.py:80-92` (union `:83`, `model_copy` `:91`) | VERIFIED | SCHEDULED. Explicit allowlist and no-widening test (ADR-SFX-004 §4) [PROPOSED] | SFX-P10-02 | SFX | specific unit (SFX-P10-02) |

### 2.2 New gaps SFX-GAP-18..27 (from W1/W2 findings, re-verified at the pin)

| Gap | Description | Evidence (path:line or doc) | Status | Disposition | Task(s) | Owner | Blocks |
|---|---|---|---|---|---|---|---|
| SFX-GAP-18 | The default `JwtAuthProvider` verifier accepts any token. Studio needs a real verifier. | `intergrax/fastapi_core/auth/providers/jwt/provider.py:27` `self._verifier = verifier or NoOpJwtVerifier()`; `verifier.py:30-40` returns empty `JwtClaims` for every token | VERIFIED | SCHEDULED. The host injects a verifier that adapts `IdentityProviderBackend.verify_token` (`integrations/contracts/identity_provider.py:44-49`), fails closed at startup, and a host test forbids NoOp (ADR-SFX-003 D3) [PROPOSED]. The platform default is unchanged by SFX and is recorded for the `fastapi_core` owner (OD-SFX-20) | SFX-P10-01 | SFX (+ platform note) | specific unit (SFX-P10-01) |
| SFX-GAP-19 | Credentialed wildcard CORS in the reference hosts **and in the `new-application` scaffold template**, which SFX-P5-06 and every `DETERMINISTIC_SCAFFOLD` output use | `applications/legal_application/host/factory.py:183-185`; `applications/local_workspace_application/host/factory.py:252-254`; template `intergrax/scaffold/new_application_product.py:625-632` (`allow_credentials=True`, `allow_methods=["*"]`, `allow_headers=["*"]`) | VERIFIED (template part is new in this consolidation) | PLATFORM_HANDOFF for the template (Tier-0, frozen surface). SFX compensations [PROPOSED]: the Studio host deviates (ADR-SFX-003 D4: no `*`, `allow_credentials=False`); SFX-P8-01 patches or flags the CORS block inside the job workspace; SFX-P9-02/P9-03 reject credentialed wildcard CORS in generated packages | SFX-P5-06, SFX-P8-01, SFX-P9-02, SFX-P10-01 | SFX + scaffold owner | specific unit (SFX-P5-06, SFX-P8-01) |
| SFX-GAP-20 | `DEAD_LETTERED` is absent from code. SFX-CAP-33 lists it as REUSE_AS_IS, which is wrong for the DLQ part. | No match for `DEAD_LETTERED` in `*.py` under `intergrax/`, `agents/`, `applications/`; PDC-D7-R013 ABSENT; ADR-PDC-07 is a decision record only (inventory `.md` "Rows with MISSING_EVIDENCE", CAP-33) | VERIFIED | EXTERNAL_DEPENDENCY (PDC-X D7). Interim per ADR-SFX-002 OQ-002-5 [PROPOSED]: report `FAILED` with `failure.code = sfx.job.retry_budget_exhausted`. CAP-33 is corrected at the inventory re-pin | SFX-P8-02; inventory re-pin | PDC-X → SFX | specific unit (SFX-P8-02 DLQ acceptance) |
| SFX-GAP-21 | No verified SFX-facing entry point into UCA (Governed Capability Fulfillment) for design-time capability-gap classes | Correction of the lane finding "UCA has no code symbol": UCA code **does** exist, scoped to worker obstacle recovery: `WorkerCapabilityAcquisitionDecisionService` (`intergrax/autonomous_work/capability_acquisition_service.py:100`), `WorkerCapabilityNeed` (`intergrax/contracts/autonomous_work/capability_acquisition.py:181`), `build_worker_capability_fulfillment_coordinator` (`intergrax/autonomous_work/worker_capability_fulfillment_composition.py:24`) [VERIFIED]. The architecture record is FROZEN (`docs/project/maintainers/architecture/GOVERNED_CAPABILITY_FULFILLMENT.md:10`) [VERIFIED]. A design-time entry point usable by SFX is not verified | MISSING_EVIDENCE | SCHEDULED [PROPOSED]: SFX-P7-01 names the entry point or limits SFX to classifying gaps and emitting proposals, with no call into the runtime worker path. CAP-05 `code_path` is corrected at re-pin | SFX-P7-01; inventory re-pin | SFX (architect) | specific unit (SFX-P7-01) |
| SFX-GAP-22 | Collaborative Work artifact semantics that SFX relies on (immutable versions, CAS publication, content store) have no SHA-pinned qualification | Inventory CAP-30 = REQUALIFICATION_REQUIRED: `PDC-D8` `qualification_suites: []`; no `tests/qualification` file references `WorkArtifactVersionRepository`, `ArtifactPublicationRepository` or `ArtifactContentStore` | VERIFIED | SCHEDULED [PROPOSED]: (a) current-HEAD replay of the CW repository qualification suite before the freeze pin; (b) SFX-P5-05 acceptance tests for append-only, CAS and tenant isolation on SQLite (+ Postgres when available) | inventory re-pin (SFX-P1-03 follow-up); SFX-P5-05 | CW domain owner + SFX | specific unit (SFX-P5-05) |
| SFX-GAP-23 | The location of "STRICT-gated MSE authorization" (R1 §14.2) was not traced; ADR-SFX-006 §6 lists it as an open item | `intergrax/applications/_shared/harness_meaningful_side_effect_authorization_wiring.py:137` `resolve_harness_host_meaningful_side_effect_authorization_wiring`: an `explicit` port wins first (`:151`); `production_mode = environment.execution_mode.value == "strict"` (`:155`); non-STRICT hosts get `authorization_port=None` (`:156-157`) | VERIFIED | RESOLVED for tracing. ADR-SFX-006 §6 row should cite `:137-157` before OD-SFX-06 [PROPOSED]. Two consequences [PROPOSED]: LAB (non-STRICT) L2 runs have **no** MSE authorization port, so the L2 profile relies on fixture adapters and egress deny (R1 §13); the `explicit` override is added to the SFX-P9-02 forbidden list (GAP-14) | ADR-SFX-006 edit; SFX-P9-02 | SFX (architect) | nothing |
| SFX-GAP-24 | A committed `web/package-lock.json` enters the IIAC G0 inventory, and excluding it needs a `scripts/` change outside SFX-P5-06 | `scripts/governance/iiac_g0_inventory.py:18-22` (all tracked `.json`), `:51` (`node_modules/` excluded only as a root prefix), `:53-57` (only `uv.lock` excluded exactly); ADR-SFX-003 D8 S-5 | VERIFIED | OD_REQUIRED (OD-SFX-05 / OD-SFX-09). Options: accept the lockfile as an enumerated neutral row (no `scripts/` change), or exclude it like `uv.lock`, which needs a new `scripts/` unit (PROPOSED ID SFX-P5-08) under C-AF-01 | SFX-P5-06, SFX-P10-04, (SFX-P5-08 if exclusion is chosen) | operator → SFX | W3 entry (via OD-SFX-05/09) |
| SFX-GAP-25 | MCP API-key callers map to one shared service principal, not to a human. MCP therefore stays at L0/L1. | `intergrax/applications/_shared/harness_auth.py:209-224` → `api_key_service_harness_principal(tenant_id=…, service_id="harness-api-key")` (`harness_principal.py:72`); identity-provider bearer maps to a user (`harness_auth.py:226-235`); local-dev bypass returns no principal when no key or provider is configured (`:185-191`, `:236-239`); ADR-SFX-004 R-004-03 | VERIFIED | OD_REQUIRED (OD-SFX-17) [PROPOSED]: L2, decisions and human-submit actions (HO-2) need a human-attributable principal (IdP bearer); API-key callers stay L0/L1; the SFX host requires an auth configuration and fails closed (no local-dev bypass outside LAB) | SFX-P10-02, SFX-P10-03 | operator → SFX | specific unit (SFX-P10-02 L2 via MCP; SFX-P10-06 approvals) |
| SFX-GAP-26 | Cursor maps `loading: explicit` rules to "Apply Intelligently" (F-01). Operator-only rules may auto-apply in Cursor. | `scripts/governance/sync_agent_env.py:100-108`; `agent-env/README.md:26`; FACTS §3.1 | VERIFIED | PLATFORM_HANDOFF (agent-env tooling owner). SFX is not affected because its rule is `loading: auto` (ADR-SFX-004 F-01). Fix candidate (omit `description` for explicit rules) is a `scripts/` change: SFX-P4-B01 scope or a separate governance ticket, under C-AF-01 and OD-SFX-20 | SFX-P4-B01 (candidate) | agent-env tooling owner | nothing (for SFX) |
| SFX-GAP-27 | Cursor discovers each generated skill twice, from `.claude/skills` and `.cursor/skills` (F-02) | `sync_agent_env.py:164-167`; FACTS §3.2 | VERIFIED (duplicate discovery) | PLATFORM_HANDOFF. Adapters point to one canonical file, so duplicates are identical (ADR-SFX-004 F-02). One-root-per-tool is an SFX-P4-B01 operator choice | SFX-P4-B01, SFX-P10-08 | agent-env tooling owner | specific unit (SFX-P10-08 live check) |
| SFX-GAP-27 (residual) | Cursor de-duplication behaviour for identical skill names | FACTS §3.2: `UNKNOWN` | MISSING_EVIDENCE | SCHEDULED (live check) | SFX-P10-08 | SFX | specific unit (SFX-P10-08) |

### 2.3 Missing-evidence item assigned to this consolidation (ADR-SFX-003 ME-3)

- **ME-3 — Phase K definition behind DX-MAINT-04 [VERIFIED].**
  - DX-MAINT-04 says "product UI belongs to Tier-3 hosts or Phase K" (`docs/project/architecture/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md:567`).
  - "Phase K" maps to product applications and agents outside the platform plan: `CONTEXT_ENGINEERING_embedded_detail.md:37` ("Phase K business agents | PLATFORM_FOUNDATION §6.3"); `docs/project/maintainers/plans/PLATFORM_FOUNDATION.md:352-356` (§6.3 "End of platform plan": product applications are out of scope).
  - **Conclusion [PROPOSED]:** Studio is product UI in a Tier-3 host, so the **Tier-3 branch** of DX-MAINT-04 applies directly. The Phase K branch is not needed. The §39.7 tension recorded in ADR-SFX-003 "Negative" is resolved. The operator confirms this in the OD-SFX-06 review.
  - **Out-of-scope finding [VERIFIED]:** `docs/project/architecture/satellites/OBSERVABILITY_extended_depth.md:213` cites `PLATFORM_FOUNDATION.md` §6.3a, which does not exist. This is recorded only, in the same way as RA-F-01..06.

### 2.4 Inventory corrections to fold into the re-pin (SFX-P1-03 follow-up)

| Row | Correction | Status |
|---|---|---|
| SFX-CAP-05 | Name the worker-scoped UCA symbols from GAP-21; the SFX entry point stays MISSING_EVIDENCE | VERIFIED (symbols) |
| SFX-CAP-19 | MCP-SERVER-X is now `CLOSED — INDEPENDENTLY AUDITED (INTERNAL AGENT CERTIFICATION) — PUBLISHED, SCOPED` (`PLATFORM_ENTERPRISE_COMPLETION_ROADMAP.md:567` at `origin/development` `c11184164`). The target class `REUSE_WITH_CONFIGURATION` becomes eligible at re-pin | VERIFIED (ledger text); the re-classification is PROPOSED |
| SFX-CAP-33 | Split: checkpoint and scheduler ledger REUSE_AS_IS; DLQ ABSENT (GAP-20) | VERIFIED |

---

## 3. Conflicts register

| ID | Conflict | Current state | Evidence | Owner / closes when |
|---|---|---|---|---|
| C-SEF-01 | SEF names no external intake consumer; CON-N-01 is "human submit only" | **OPEN** [BLOCKED for SFX-P8-03]. SFX conforms through HO-2 | `ASF-SEF-MASTER-PLAN.md:381`; ADR-SFX-006 §9 | SEF track (OD-SEF-07 or a new SEF OD) + OD-SFX-10 |
| C-SEF-02 | SEF T-P5-004 (network egress) is OPTIONAL and excluded from certification | **OPEN** [BLOCKED for GATE-SFX-FULL] | `ASF-SEF-MASTER-PLAN.md:516`, `:588`; GAP-MED-07 `:275` (ADR-SFX-006 §9) | SEF track makes T-P5-004 mandatory for GATE-SEF-CONSUME |
| C-SEF-03 | CON-E-04 `VerificationRecord` has no SEF WBS task | **OPEN** [BLOCKED for SFX consumption of CON-E-04] | `ASF-SEF-MASTER-PLAN.md:372` | SEF track assigns a task |
| C-SEF-04 | GAP-HIGH-07 (SEF ↔ READINESS kit coupling) is OPEN | **OPEN**. SFX stays on the existing scaffold tools and never defines kit templates [PROPOSED rule, ADR-SFX-006 §9] | `ASF-SEF-MASTER-PLAN.md:264` | SEF + READINESS tracks; READINESS kit CLOSED |
| C-RDY-01 | SFX must not define Delivery Kit templates; REF-2 must not duplicate the READINESS settlement reference | **OPEN** (guard rule in force) [PROPOSED]. Re-checked at OD-SFX-08 (REF-2/REF-3 inclusion) | R1 §12, §16 | Operator at OD-SFX-08; READINESS-X track |
| C-AF-01 | `scripts/` changes interact with the ARCH-FREEZE inventory and SHA lock | **OPEN**, with concrete instances [VERIFIED]: SFX-P4-B01 (`sync_agent_env.py`), the possible IIAC lockfile exclusion (GAP-24), the F-01 generator fix (GAP-26), and OQ-002-2 (conformance entry point under `scripts/`) | R1 §15.1; `iiac_g0_inventory.py:53-57`; `sync_agent_env.py:100-108` | Operator: OD-SFX-12 (pre-lock pull-in) or default W3 placement |
| C-IIAC-01 | Every post-freeze product wave needs an ACT-N11 audited re-bind (REBIND-Wn) | **OPEN** [VERIFIED as a standing obligation: IIAC IN_FORCE, ACT-N11 prefixes `intergrax/`, `agents/`, `applications/`, R1 §1] | R1 §10 "Post-freeze waves", §16 | OD-SFX-09 sets the IIAC scope; each REBIND-Wn |
| C-ECAP-01 | Does ECAP-§16 apply to a Tier-2 product module? | **PROPOSED resolution:** not applicable under Option B; no ECAP revision; optional closure note (ADR-SFX-001 "C-ECAP-01") | `ENTERPRISE_CONTINUOUS_ARCHITECTURE_ASSURANCE_PROTOCOL.md:1109-1112`, `:3-4` (cited in ADR-SFX-001) | Operator in the OD-SFX-06 review of ADR-SFX-001 |
| C-OD-01 | Do the regenerated `.claude/` and `.cursor/` adapters count as an agent-env delivery before the freeze? | **RESOLVED by OD-SFX-16** ("Yes, with agent-env"; `sync_agent_env.py` and `role_registry.yaml` untouched) [VERIFIED: `DELEGATION-SFX-PRE-FREEZE.md` item 3; signing commit `4758c4051` on `sfx-p0-registration`]. Publication to `development` is pending with T-IGX-037-A [MISSING_EVIDENCE until merged] | `docs/project/maintainers/qualification/scenario_factory_x/operator_decisions/DELEGATION-SFX-PRE-FREEZE.md:26` | Closed. Only the W0 merge remains |

---

## 4. Operator decision backlog

All recommendations are **PROPOSED**. The operator (or an explicitly delegated session) decides. OD signatures are not delegated by `DELEGATION-SFX-PRE-FREEZE`.

### 4.1 R1 decisions OD-SFX-05..14

| OD | Question | Options | Recommended | Blocking unit |
|---|---|---|---|---|
| OD-SFX-05 | Node toolchain, package manager, lockfile and npm audit policy for `web/`, including the IIAC treatment of the lockfile (GAP-24) | (a) Node LTS + npm + committed `package-lock.json`; lockfile as an enumerated IIAC neutral row. (b) The same, with the lockfile excluded like `uv.lock` (new `scripts/` unit). (c) pnpm/yarn | (a): no `scripts/` change, no C-AF-01 exposure. ME-5 (Windows run) is proven in SFX-P10-04 | **GATE-SFX-ENTRY**; SFX-P5-06, SFX-P10-04 |
| OD-SFX-06 | Accept ADR-SFX-001..006, including C-ECAP-01, ME-1, ME-2, ME-3, OQ-002-1..6, UX-Q-04/05 | Accept / accept with amendments / reject per ADR | Accept after (1) an independent reviewer session (SFX-P2-01 acceptance), (2) a clean `check_harness_adr.py` run, (3) the GAP-23 citation edit in ADR-SFX-006 | **GATE-SFX-ENTRY** |
| OD-SFX-07 | TTFS, usability (SUS, task success) and G-1..G-5 targets (= UX-Q-07) | Set targets at P3 exit / defer to the round-0 baseline | Set them at P3 exit from the round-0 baseline (`SFX_USABILITY_PROTOCOL.md`) | SFX-P6-01 acceptance threshold, SFX-P12-04, SFX-P12-05 |
| OD-SFX-08 | REF-2 / REF-3 inclusion (= UX-Q-08) | Include both / REF-1 only / REF-1 + REF-3 | Include both, with the C-RDY-01 guard on REF-2 | SFX-P12-02/03; usability rounds 2–3 |
| OD-SFX-09 | IIAC scope for SFX (per-stage activation, REBIND-Wn, lockfile row) | Per-wave activation / single stage activation at W3 | Per-wave REBIND-Wn as in R1 §10, with lockfile handling decided together with OD-SFX-05 | **GATE-SFX-ENTRY**; every W3+ wave |
| OD-SFX-10 | Cross-track edge SFX → SEF-X; SFX as a registered SEF intake consumer | Sign now / sign when SEF reaches GATE-SEF-FULL | Sign now as a one-way edge (SEF does not depend on SFX), so SEF can record C-SEF-01 early | GATE-SEF-CONSUME; SFX-P8-03 |
| OD-SFX-11 | SFX stage closure | — | At W8 after SFX-P13-03 | SFX-P13-04 |
| OD-SFX-12 (optional) | Pull SFX-P4-B01/B02 before the freeze as Class A tooling (only before `T-B7-LOCK-FREEZE-SHA`) | Pull in / keep in W3 | Keep in W3. v0 works without native agents (FACTS §5.3), and keeping it avoids a `scripts/` change near the SHA lock (C-AF-01) | SFX-P4-B01/B02 |
| OD-SFX-13 | Owner of the per-scenario production admission record | SCENARIO-GATE programme / Agent Runtime Governance | Agent Runtime Governance (it already owns the S1–S6 seams and the roster certification path) | Production admission only |
| OD-SFX-14 | Adopt SAST and dependency tools (semgrep, ruff `S`, pip-audit, npm audit) in the PR pipeline | (a) SFX-scoped jobs only (generated packages + `web/`). (b) Repository-wide PR gates. (c) Release-only, as today | (a) first: it covers SFX risk without changing repository-wide gates | **GATE-SFX-ENTRY**; SFX-P9-03, SFX-P10-04 |

### 4.2 New decisions (PROPOSED IDs) and UX-Q mapping

| ID | Question | Options | Recommended | Blocking unit |
|---|---|---|---|---|
| **OD-SFX-17** (UX-Q-01 + GAP-25) | Which doors may record approval decisions and human submits? What does an API-key MCP caller map to? | (a) Studio-only inbox, as an explicit PC-6 exception. (b) Every door, but only with a human-attributable principal (IdP bearer); API-key callers L0/L1. (c) Every door with any principal | (b): keeps PC-6 1:1 and HO-2 attributable | **Recommended before W3** (it shapes CON-13 approval and the PC-6 op map in SFX-P5-03); SFX-P10-02, SFX-P10-06 |
| **OD-SFX-18** (UX-Q-02 + UX-Q-03) | Is self-approval forbidden? Is the dossier export (`SANDBOX_CERTIFIED → ADMISSION_DOSSIER_READY`) a human-decision gate, and who decides? | Self-approval: forbid / allow. Dossier: approver persona / separate role / automatic | Forbid self-approval (requester ≠ approver, enforced in the service layer). The dossier export is a human decision by an approver who is neither the requester nor the independent evaluator. Recorded as an ADR-SFX-006 amendment | **Recommended before W3** (SFX-P5-03 CON-13/CON-17); SFX-P11-04 |
| **OD-SFX-19** (UX-Q-06) | Is document upload on Describe in v1 scope, and which data classes are allowed in the sandbox? | No upload in v1 / synthetic-only upload / classified upload with retention | Synthetic-only upload in v1, behind the SFX-P9-05 injection suite; no personal data in LAB | SFX-P10-05, SFX-P9-05 |
| **OD-SFX-20** (platform handoff) | How are the platform-owned findings handed over while ARCH-FREEZE is still OPEN: S1–S6 (GAP-15), the NoOp verifier default (GAP-18), the scaffold CORS template (GAP-19), F-01/F-02 (GAP-26/27)? | (a) Record them now as findings to PDC-X D1/D6 and the scaffold / `fastapi_core` / agent-env owners, docs only. (b) Defer them to SFX closure | (a): they are platform scope and cheaper to settle before the freeze lock. SFX changes none of them | Nothing in SFX; production admission (S1–S6) |
| UX-Q-04 → ADR-SFX-004 §3 | How do offline L1 files enter the shared store? | Import operation through an authenticated door / live mode only | An `import` operation on the API and MCP doors that re-validates and re-canonicalizes. The CLI never writes shared stores | Decided at OD-SFX-06; SFX-P5-07, SFX-P10-03 |
| UX-Q-05 → ADR-SFX-002 D6/D7 | Does any configuration change after APPROVED invalidate the approval? | Any digest change invalidates / semantic-diff threshold | Any blueprint digest change invalidates (no `APPROVED → BLUEPRINTED` edge) | Decided at OD-SFX-06; SFX-P5-05, SFX-P7-04 |
| UX-Q-07 / UX-Q-08 | — | — | Map to OD-SFX-07 / OD-SFX-08 | as above |
| OQ-002-1..6 | ADR-SFX-002 open questions | Per ADR | Per the ADR's proposed answers | Decided at OD-SFX-06 |
| ME-2 | Does adopting a JS toolchain need its own ADR? | ADR-SFX-003 covers it / dedicated ADR | ADR-SFX-003 covers it | Decided at OD-SFX-06; SFX-P10-04 |

---

## 5. GATE-SFX-ENTRY readiness checklist (input to PRE_IMPLEMENTATION_READY)

### 5.1 Gate conditions (R1 §10 "Gate: GATE-SFX-ENTRY")

| # | Condition | Current state | Status | What closes it |
|---|---|---|---|---|
| E-1 | ARCH-FREEZE CLOSED | OPEN. At `origin/development` `c11184164`, MCP-SERVER-X is CLOSED (scoped) and PDC-X is `PLANNED / MANDATORY / CURRENT` (`PLATFORM_ENTERPRISE_COMPLETION_ROADMAP.md:567`). ARCH-FREEZE needs MCP-SERVER-X and PDC-X CLOSED (`:241`, `:497`) | BLOCKED | PDC-X closure, then ARCH-FREEZE certification with `FREEZE_CERTIFICATION_SHA` |
| E-2 | ADR-SFX-001..006 Accepted (OD-SFX-06) | All six exist with Status **Proposed** (line 5 of each) [VERIFIED]. Still to do: the GAP-23 edit in ADR-SFX-006; an independent reviewer session; a `check_harness_adr.py` run (not run by this lane: MISSING_EVIDENCE); operator decisions on C-ECAP-01, ME-1, ME-2, ME-3, OQ-002-1..6, UX-Q-04/05 | PROPOSED | T-IGX-037-C test contract + reviewer session + OD-SFX-06 signature |
| E-3 | OD-SFX-05 signed | No OD-SFX-05 record exists (`operator_decisions/` in `sfx-p0` holds OD-SFX-01..04, 15, 16 only) [VERIFIED] | MISSING_EVIDENCE | Operator signature (including the GAP-24 lockfile choice) |
| E-4 | OD-SFX-09 signed | Not recorded [VERIFIED absence] | MISSING_EVIDENCE | Operator signature |
| E-5 | OD-SFX-14 signed | Not recorded [VERIFIED absence] | MISSING_EVIDENCE | Operator signature |
| E-6 | Inventory re-pinned at `FREEZE_CERTIFICATION_SHA` | Inventory pinned at `eadb1c21` (37 rows) [VERIFIED]. Product code unchanged to `c11184164` [VERIFIED]. `FREEZE_CERTIFICATION_SHA` is TBD | BLOCKED (on E-1) | Re-pin run with the §2.4 corrections and the CW requalification replay (GAP-22) |
| E-7 | GEN set sealed | `SFX-GEN-SEAL-01` exists with `GEN_SET.json` digest `8b6ff8eb…3e6e0d`, authored by the independent session `sfx-gen-author` [VERIFIED]. It is untracked in worktree `sfx-w2-ux` (not committed or published); `GEN_SEAL_SHA` is unset | PROPOSED | T-IGX-037-D merge and the `GEN_SEAL_SHA` ladder row |

OD-SFX-13 is **not** an entry condition. It gates admission only (R1 §10).

### 5.2 Pre-implementation prerequisites (W0–W2 deliverables; not in the R1 gate list, but needed for PRE_IMPLEMENTATION_READY)

| # | Item | Current state | Status | What closes it |
|---|---|---|---|---|
| P-1 | W0 T-IGX-037-A published | `Approved:` delegate line present (`T-IGX-037-A.md:24`); OD-SFX-01..04, 15, 16 signed (commit `4758c4051`); W0 docs (roadmap, SSOT plan, dossier skeleton) uncommitted in `sfx-p0` [VERIFIED] | PROPOSED | Commit, PR, review session, operator merge (R1 §15.3) |
| P-2 | W1 T-IGX-037-B complete | Inventory and IDE facts present; competitive matrix SFX-P1-01 absent (GAP-11) | MISSING_EVIDENCE | SFX-P1-01 delivered, then the T-IGX-037-B test contract and merge |
| P-3 | W1 T-IGX-037-C complete | ADRs (6), schemas (17 files), `SFX_CANONICALIZATION.md`, `SFX_METRICS.md` and this file present, all untracked in `sfx-w1-arch` [VERIFIED]; the test contract has not been run | PROPOSED | `check_harness_adr.py`, `pytest tests/unit/governance -q`, `igx_gov ticket`; review; merge |
| P-4 | W2 T-IGX-037-D complete | `SFX_UX_DESIGN.md`, `SFX_USABILITY_PROTOCOL.md`, `SFX_GEN_SET_SEAL.md` present, untracked [VERIFIED]; usability round 0 result: MISSING_EVIDENCE | PROPOSED | Round-0 report, review, merge |
| P-5 | W2 T-IGX-037-E (IDE v0) complete | `sfx-w2-ide` shows only untracked `agent-env/agents/` [VERIFIED]; rule, skills and regenerated adapters not seen | MISSING_EVIDENCE | SFX-P4-A01/A02 delivered with `sync_agent_env.py --check` clean |
| P-6 | Decisions recommended before W3 | OD-SFX-17 and OD-SFX-18 shape SFX-P5-03 contracts; OD-SFX-07/08 are due at P3/P2 exit; OD-SFX-10 is best signed early | PROPOSED | Operator decisions |
| P-7 | C-OD-01 closed | Resolved by OD-SFX-16 [VERIFIED] | VERIFIED | — (publication with P-1) |
| P-8 | GAP-23 open item in ADR-SFX-006 closed | Traced in this file [VERIFIED] | VERIFIED | ADR-SFX-006 §6 citation edit before OD-SFX-06 |

### 5.3 Top blockers for GATE-SFX-ENTRY (ordered)

1. **E-1 ARCH-FREEZE OPEN** (via PDC-X). This is outside SFX control and also gates E-6 [BLOCKED].
2. **E-2 ADR acceptance:** reviewer session, ADR check run, OD-SFX-06 [PROPOSED].
3. **E-3..E-5 OD-SFX-05/09/14 unsigned.** OD-SFX-05 and OD-SFX-09 must settle the lockfile treatment together (GAP-24, C-AF-01) [MISSING_EVIDENCE].
4. **E-7 GEN seal unpublished** [PROPOSED].
5. **P-2 / P-5:** the competitive matrix and IDE v0 deliverables were not found [MISSING_EVIDENCE].
