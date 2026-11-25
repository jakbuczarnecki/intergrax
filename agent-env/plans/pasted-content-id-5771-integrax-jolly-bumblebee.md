# SCENARIO-FACTORY-X (SFX) — Master Campaign Plan · Revision R1 (corrective review)

> **Status:** plan for operator approval. It is planning only; no product code.
> **Scope:** R1 corrects R0 (approved 2026-11-12) along the 12 operator findings. The R1 deltas are listed in §17 `CORRECTION_MATRIX`.
> **Unchanged from R0:** Option B architecture, product scope, two front doors (Scenario Studio + IDE toolkit), roadmap position, and OD-SFX-01..04.
> **Language:** repository text is English; the operator chat is Polish.
> **Approval scope:** approving R1 approves only **W0–W2** (pre-freeze, as classified in §15.1). Waves W3+ need ARCH-FREEZE CLOSED, GATE-SFX-ENTRY and their own ticket approvals. **Approving this plan is not an authorization for post-freeze implementation.**

## Context

The goal is unchanged from R0. Intergrax should become an Enterprise Scenario Factory: a business problem goes in, and a discoverable, configurable, executable, testable and demonstrable scenario comes out. It must be built only from existing platform mechanisms. That rules out a second engine, a second governance authority, a second ToolRuntime, a second registry and a second audit mechanism.

**Operator decisions in force (English rendering of the Polish chat, 2026-11-12):**

| ID | Decision |
|---|---|
| **OD-SFX-01 (placement)** | SFX is a new, standalone stage and the first product priority after ARCH-FREEZE. SFX is the product layer: Studio, Discovery, Specification, Blueprint, Capability Resolution, Profile Composition, Integration Setup, Registry, Evaluation, Reporting, ROI. SEF-X provides the engineering engine; SFX uses it and does not duplicate it. READINESS-X provides the Delivery Kit; SFX uses it and does not take it over. Intergrax Runtime remains the only execution environment. SFX implementation requires ARCH-FREEZE CLOSED. SFX does not block READINESS-X. OD-SEF-02 parallelism is preserved. SFX may progress in parallel with SEF-X, but SEF-dependent features need accepted SEF-X contracts. SFX does not replace SCENARIO-GATE or PROD-Q. |
| **OD-SFX-02 (stack)** | React + TypeScript + Vite SPA in the repo, under the Tier-3 host `applications/scenario_studio_application/web/`, with a FastAPI backend on `intergrax/fastapi_core`. |
| **OD-SFX-03 (IDE model)** | One factory core. agent-env skills, rules and agents use the CLI offline, with an optional MCP live mode (streamable HTTP). |
| **OD-SFX-04 (pre-freeze)** | Before ARCH-FREEZE CLOSED, only these are allowed: docs, the roadmap amendment, cited research, contract/UX design and mock-ups, and IDE toolkit v0 **in agent-env only**, with proposal-only outputs. |

R1 changes no OD. Where R0 conflicted with an OD, R1 brings the plan back to the OD (see §15.1, C-OD-01).

---

## 1. Current HEAD and predecessor readiness (unchanged facts, VERIFIED @ `origin/development` `eadb1c2116fe`)

| Programme | State |
|---|---|
| MCP-SERVER-X | P7 package @ `abbe15254`; P8 in progress; audit and closure SHAs unset |
| PDC-X | P0–P4 published; wave-1 tickets 036-J/K/L `plan_approved`; product merges wait for MCP P8 |
| ARCH-FREEZE | **OPEN**: 54 tasks (18 complete, 25 planned, 10 blocked-external, 1 superseded); `FREEZE_CERTIFICATION_SHA` TBD |
| SEF-X | docs only (P1 closed); 0 code for CAP-049..052 and the CON-N/CON-E contracts; ADR-SEF-001 **Proposed**; OD-SEF-04..08 PENDING |
| READINESS-X | PLANNED, not started; owns the Scenario Delivery Kit analysis |
| SCENARIO-GATE | BLOCKED; a **programme** Go/No-Go (see §14.2) |
| PROD-Q live slice | NOT QUALIFIED (DIM6/DIM17 OPEN) |
| FRZ | 225 / 26 families (recomputed) |
| IIAC | IN_FORCE; ACT-N11 product prefixes `intergrax/`, `agents/`, `applications/`; material governance paths are the 5 IIAC scripts plus the protocol .md |

Out-of-scope findings RA-F-01..06 are unchanged from R0. They are recorded and not fixed by SFX. RA-F-02 (ledger "Next" and the missing SEF-X row) is still fixed in the amendment.

**Main checkout `D:\Projekty\intergrax`** is in active use by another session: it moved from `mcp-p7-qualification` to `mcp-p8-audit-evidence` during this planning session. SFX never writes there; see §15.3.

---

## 2. Roadmap amendment (A1–A10 kept; R1 wording corrections marked ▲)

The positions, the structure and the "no predecessor change" rule are unchanged from R0. Edits are anchored on text, not on line numbers.

- **A1 ▲ (header L7):** "`SCENARIO-FACTORY-X` (§3.0.7b) is post-freeze **product tooling**. Scenarios it produces stay proposal-only or sandbox (LAB). Production use of any generated scenario requires the scenario programme opened by `SCENARIO-GATE`, a per-scenario production admission by its designated owner, and per-integration `PROD-Q` live qualification. **SFX stage closure admits no scenario to production.**"
- **A2 (§2.0.2 FRZ linkage row):** CTR, OWN, BND, TYP, GOV, SEC, TEN, TRC, OBS, CFG, EVOL, DOC. These are incremental evidence; there are no `FRZ-SFX-*` IDs and the count stays 225.
- **A3 (§2.0.3 predecessors row):** unchanged.
- **A4 ▲ (§3 row between ARCH-FREEZE and SEF-X):** the R0 text with three replacements:
  - "Consumes SEF-X engineering capability" becomes: "**two engineering modes:** `DETERMINISTIC_SCAFFOLD` (existing scaffold tools) and `AGENTIC_ENGINEERING` (SEF-X EEA). **SFX closure requires the agentic mode to be verified end-to-end on Accepted and certified SEF-X mechanisms.**"
  - "registry" becomes "catalog (metadata sidecar; no new registry)".
  - Add: "SFX product certification ≠ admission of any generated scenario."
  - Status stays `[ ] PLANNED / MANDATORY / POST-ARCH-FREEZE PRIORITY`.
- **A5 (ARCH-FREEZE next-stages sentence):** unchanged.
- **A6 ▲ (§3.0.7b subsection):** adds:
  - the two-mode engineering rule;
  - the dependency "SFX closure ← SEF-X GATE-SEF-FULL + SEF scoped B6 (T-P12-003) for consumed mechanisms". This is a documented one-way cross-track edge; SEF-X and READINESS do not depend on SFX.
  - the admission separation (§14).
- **A7 (chain blocks):** `ARCH-FREEZE → SCENARIO-FACTORY-X ∥ SEF-X ∥ SCENARIO-READINESS-X → SCENARIO-GATE`, plus the footnote "SFX is not a SCENARIO-GATE predecessor; SFX closure follows SEF-X engineering certification".
- **A8 (§5 ledger):** adds the SFX row and the SEF-X row (RA-F-02), and corrects ARCH-FREEZE "Next".
- **A9 (§7 goal bullet):** unchanged.
- **A10 (amendment record):** unchanged.
- **Regression:** `tests/qualification/roadmap_replay_x` parses the §3 and §5 cell grammar. Compare against the red baseline at `eadb1c21`.

---

## 3. SFX ↔ SEF-X relationship (▲ corrected)

### 3.1 Two engineering modes behind one `EngineeringHandoffPort`

| | `DETERMINISTIC_SCAFFOLD` | `AGENTIC_ENGINEERING` |
|---|---|---|
| What | Template fill from an approved blueprint, using `python -m intergrax.scaffold new-application/new-agent/new-integration/new-skill` and `scripts/proof/init_scenario_implementation.py`. No LLM. Same blueprint digest gives the same tree digest. | Implementation of business logic, tests and evidence by the SEF-X Engineering Execution Authority (EEA). |
| Output | A conformant skeleton (structure, contracts, wiring, test stubs). Business logic is `NotImplemented`. | A complete scenario package plus a PR, `EngineeringEvidenceBundle` (CON-E-05) and `DeliveryManifest` (CON-N-04). |
| Authority | SFX handoff adapter writes only inside its job workspace (§13). | SEF-X: `WorkspaceLease` (CON-N-03), `CodingExecutionBackend` (CON-N-05), engineering FSM, DevOS ticket and PR, no auto-merge (T-P9-005). |
| Available | After GATE-SFX-ENTRY (W4). | Only after **GATE-SEF-CONSUME** (§3.3). |
| Certifies | Factory determinism and conformance. | **Required for SFX closure** (GATE-SFX-FULL): REF-1..3 plus the generalization set (§8). |
| Fallback | — | The deterministic scaffold may still be used. A scaffold-only result can never reach SANDBOX_CERTIFIED with business logic. |

### 3.2 Certifiable end-to-end engineering handoff (SFX → SEF-X → SFX)

```mermaid
sequenceDiagram
  participant U as Approver (human)
  participant SFX as SFX handoff (Tier-2)
  participant SEF as SEF-X EEA
  participant DevOS as DevOS ticket/PR
  participant Q as SFX qualification
  SFX->>SFX: blueprint vN APPROVED (decision binding, digest D_bp)
  SFX->>U: EngineeringIntent draft (CON-N-01 fields; text_untrusted ⊇ canonical blueprint, D_bp)
  U->>SEF: submit intent (operator_id = approver)  %% SEF rule: human submit only
  SEF->>DevOS: EngineeringSpecification (CON-E-01) → ticket + plan digest → awaitingPlanApproval
  DevOS-->>SEF: Approved (operator/delegate, not SFX)
  SEF->>SEF: WorkspaceLease → executing → verifying (VerificationRecord CON-E-04) → evidenceAssembling
  SEF-->>SFX: EngineeringExecutionRecord (CON-N-02) state + EngineeringEvidenceBundle (CON-E-05) + DeliveryManifest (CON-N-04)
  SFX->>Q: verify bundle digest, D_bp lineage, package conformance, SFX gates (§9, §13)
  Q-->>SFX: ScenarioQualificationRecord (sandbox) → SANDBOX_TESTED; independent check → SANDBOX_CERTIFIED
```

**Handoff certification criteria (HO-1..HO-7):**
- **HO-1** Lineage: the bundle references `D_bp`, and every artifact digest in `DeliveryManifest` is reproducible.
- **HO-2** Only a human submits an intent; SFX never auto-submits.
- **HO-3** The SFX job record links to the SEF `job_id` by reference only. SFX does not copy the SEF state machine.
- **HO-4** The generated package passes `scenario_architecture_conformance` and the SFX gates (§9).
- **HO-5** Failure states map one-to-one: SEF `failed` / `paused` become the SFX job's `FAILED` / `WAITING`, and there are no silent retries.
- **HO-6** Merge stays operator-only.
- **HO-7** The evidence is pinned to exact SHAs and enters the SFX SHA ladder.

### 3.3 GATE-SEF-CONSUME (precondition for `AGENTIC_ENGINEERING`)

All of the following must hold:
- ADR-SEF-001 is **Accepted**.
- OD-SEF-04 (FSM ↔ ticket binding) is resolved.
- CON-N-01/02/03/04/05 and CON-E-01/02/04/05 are implemented.
- SEF **GATE-SEF-FULL** has passed.
- SEF scoped B6 (T-P12-003) has passed on the consumed mechanisms.
- OD-SFX-10 (cross-track edge) is signed.
- SEF records SFX as a registered intake consumer (C-SEF-01).

### 3.4 Conflicts with SEF-X (reported, not resolved unilaterally)

- **C-SEF-01:** SEF names no external intake consumer, and its CON-N-01 is "human submit only". SFX conforms to this through HO-2. SEF must still acknowledge SFX as a consumer, under OD-SEF-07 or a new SEF OD.
- **C-SEF-02:** SEF T-P5-004 (network egress) is OPTIONAL and draft. SFX closure requires network restriction on agentic engineering (§13), so SFX asks SEF to make T-P5-004 mandatory for GATE-SEF-CONSUME. Otherwise GATE-SFX-FULL stays BLOCKED.
- **C-SEF-03:** CON-E-04 `VerificationRecord` has no SEF WBS task. SFX consumes it, so it is BLOCKED until SEF assigns one.
- **C-SEF-04:** GAP-HIGH-07 (SEF ↔ READINESS kit coupling) is OPEN. SFX scaffold templates stay on the existing scaffold tools until the READINESS kit is CLOSED; SFX never defines kit templates.

---

## 4. Competitive research — plan and evidence matrix (unchanged; findings MISSING_EVIDENCE, scheduled for SFX-P1-01)

**Products:** n8n, Copilot Studio, Dify, Flowise, Langflow, Retool, Zapier, Make, plus AI-native builders (OpenAI Agent Builder/AgentKit, Google Vertex Agent Builder, AWS Bedrock Agents, Relevance AI, Vellum, Gumloop, StackAI). Each AI-native builder's inclusion is confirmed by a primary source.

**Interaction models:** conversational, wizard, canvas, hybrid, schema-driven.

**Dimensions:** D-01..D-14, as in R0: discovery, generation, configuration, integrations, HITL, secrets, testing, versioning, deployment, results, ROI, usability/TTFS, production readiness, IDE parity.

**Rules:** only vendor primary documentation with URL and access date; anything unsupported is `UNKNOWN`. Output goes to `…/satellites/scenario_factory/SFX_COMPETITIVE_EVIDENCE_MATRIX.md`.

**IDE/MCP facts (SFX-P1-02):** the Claude Code `.claude/agents` format; the Cursor rules/skills/agents format; streamable-HTTP MCP client support in both.

---

## 5. Closed-world capability inventory (R0 seed kept; ▲ rows changed)

Schema and method are as in R0, with re-pinning at `FREEZE_CERTIFICATION_SHA`. Changed and added rows:

| CAP | Owner / path (VERIFIED) | SFX use | REUSE_CLASS |
|---|---|---|---|
| SFX-CAP-08 ▲ | Durable HITL: `intergrax/collaborative_work/repository.py:1414` `CollaborativeDecisionBindingRepository` (in-memory, SQLite, Postgres) and `intergrax/runtime/human/persistence_contract.py:16` `HumanDecisionPersistence` (SQLite) | approvals | REUSE_WITH_CONFIGURATION. `ApprovalStorePort` / `InMemoryApprovalStore` is **not used**: in-memory only, no tenant parameter. |
| SFX-CAP-12 ▲ | `ApplicationManifest` (`manifest.py:319`, `extra="forbid"`, no metadata slot), `ApplicationPackage` (`application_package.py:56`) | referenced by digest only | REUSE_AS_IS. **No EXTEND.** |
| SFX-CAP-30 ▼new | Collaborative Work artifacts: `WorkArtifactRepository` L1284, `WorkArtifactVersionRepository` L1311 (immutable, append-only), `ArtifactPublicationRepository` L1338 (CAS), `ArtifactContentStore` (`content_storage.py:105`, sha256, tenant+workspace) | specs, blueprints, reports, catalog sidecar, lifecycle records | REUSE_WITH_CONFIGURATION. ADR-PDC-08 business-artifact class. Qualification status MISSING_EVIDENCE, so REQUALIFICATION_REQUIRED. |
| SFX-CAP-31 ▼new | `WorkItemRepository` L893, `AssignmentRepository` L925, `WorkItemExecutionLinkRepository` L1014 | one WorkItem per scenario; engineering jobs | REUSE_WITH_CONFIGURATION |
| SFX-CAP-32 ▼new | `SessionManager` (`runtime/nexus/session/session_manager.py:44`) + `SessionStorage` (SQLite / DocumentStore); `ChatSession.metadata` | resumable sessions | REUSE_AS_IS. Retention via MemoryControlPlane (ADR-PDC-02). |
| SFX-CAP-33 ▼new | Long-running `TaskCheckpointPersistence` / `SchedulerLedger` (SQLite, Postgres); `DEAD_LETTERED` (ADR-PDC-07) | scaffold jobs, retries, DLQ | REUSE_AS_IS |
| SFX-CAP-34 ▼new | `EvidencePersistencePort` (`contracts/execution_evidence/persistence_port.py:23`) | run facts for reports | REUSE_AS_IS |
| SFX-CAP-35 ▼new | `AgentCatalogEntry` (`agent_distribution/catalog.py:99`, "catalog metadata only") | **pattern** for the scenario catalog sidecar | pattern reuse (no import of agent semantics) |
| SFX-CAP-36 ▼new | `intergrax/contracts/sandbox_network_egress.py` (typed allowlist); `runtime/sandbox/isolation_gate.py` (fail-closed presence check) | L2 network restriction | REUSE_WITH_CONFIGURATION. Enforcement strength depends on the provider (MISSING_EVIDENCE). |
| SFX-CAP-37 ▼new | Release-only SAST (`scripts/gates/check_harness_security_promote_gate.py`: trivy and semgrep, run in `harness-release.yml`) | generated-code scanning | EXTEND usage (tool adoption per OD-SFX-14) |
| SFX-CAP-26 ▲ | SEF CON-N-01..05, CON-E-01/02/04/05 | `AGENTIC_ENGINEERING` | BLOCKED until GATE-SEF-CONSUME |

---

## 6. Gap register (▲ revised)

| Gap | Status | Disposition / task |
|---|---|---|
| SFX-GAP-01 No web toolchain, browser auth or SPA serving | VERIFIED | §7.8. Tasks SFX-P10-04..07; OD-SFX-05 |
| SFX-GAP-02 No scenario catalog | VERIFIED | Tier-2 sidecar `sfx.scenario_catalog_entry.v1` (§7.6). No manifest change. SFX-P11-03 |
| SFX-GAP-03/04 Discovery agent and ROI component absent | VERIFIED | SFX-P6-01, SFX-P11-02 |
| SFX-GAP-05 SEF contracts PROPOSED_ONLY | VERIFIED | GATE-SEF-CONSUME (§3.3) |
| SFX-GAP-06 `sync_agent_env.py` generates no agents | VERIFIED | SFX-P4-B01 (post-freeze by default; §15.1) |
| SFX-GAP-07 MCP client compatibility | MISSING_EVIDENCE | SFX-P1-02, SFX-P10-02 |
| SFX-GAP-08 ▲ `ApprovalStorePort` is in-memory with no tenant | VERIFIED | Avoided by using decision bindings (CAP-08) |
| SFX-GAP-09 Dual `ToolRegistry` | MISSING_EVIDENCE | SFX-P2-01 ownership review |
| SFX-GAP-10 Business-scenario workspace location | VERIFIED | ADR-SFX-002 |
| SFX-GAP-11 Research facts | MISSING_EVIDENCE | SFX-P1 |
| SFX-GAP-12 No SFX qualification suite | VERIFIED | SFX-P9, SFX-P13 |
| SFX-GAP-13 ▼new No PR-level SAST or dependency audit; ruff has no `S` rules; no pip-audit or npm audit | VERIFIED | SFX-P9-03; OD-SFX-14 |
| SFX-GAP-14 ▼new No conformance rule forbids generated code from mutating environment profile, policy mode, governance or certification records, contract lifecycle, or exposure policy | VERIFIED | SFX-P9-02 |
| SFX-GAP-15 ▼new Platform self-promotion seams: `AgentContract.lifecycle_state` defaults to PRODUCTION (`agent_contract_meta.py:71`); environment profile not frozen (`root.py:85`); `PRODUCTION_ATTACHED` does not require STRICT+PRODUCT; `evaluate_agent_promotion` not wired; `apply_roster_agent_governance` self-certifies | VERIFIED | **Platform-owned** (Agent Runtime Governance / PDC-X D1/D6), not fixed by SFX. SFX compensates inside its own scope (§13), records the seams and hands them to the platform owner. They block **production admission**, not SFX closure. |
| SFX-GAP-16 ▼new No owner for per-scenario production admission | VERIFIED | OD-SFX-13 (§14.2) |
| SFX-GAP-17 ▼new MCP composition widens `allowed_binding_ids` (`mcp_application_server.py:83-91`) | VERIFIED | SFX MCP uses an explicit allowlist with a no-widening test (SFX-P10-02) |

---

## 7. Target architecture (Option B kept; ▲ refined)

### 7.1 Options

Same as R0: A (core domain) rejected; **B preferred**; C (external) rejected; D (IDE-only) is a subset.

### 7.2 Component view

```mermaid
flowchart TB
  subgraph FD[Front doors — adapters only]
    ST[Studio SPA React+TS]
    IDE[Claude Code / Cursor skills+agents]
  end
  subgraph T3[Tier-3 applications/scenario_studio_application]
    API[Studio API v1]
    CLI[sfx CLI]
    MCP[MCP surface explicit allowlist]
  end
  subgraph T2[Tier-2 agents/scenario_factory]
    SVC[Service layer = single semantic implementation]
    CON[Contracts + JSON Schemas + canonicalizer]
    AG[Discovery / Spec agents — STAGING, NexusLoop]
    RES[Resolver] --- BP[Blueprint/Profile composer] --- INT[Integration setup]
    HO[EngineeringHandoffPort<br/>DETERMINISTIC_SCAFFOLD · AGENTIC_ENGINEERING]
    QUAL[Scenario qualification + security gates]
    REP[Reporter / ROI] --- CATS[Catalog sidecar]
    PERS[Persistence adapters → existing owners]
  end
  subgraph P[Frozen platform — reuse only]
    EXE[execute_scenario_task → HostTaskExecutionPort → NexusLoop → PolicyEngine → ToolRuntime]
    CW[Collaborative Work: WorkItem · WorkArtifact(Version) · DecisionBinding · ContentStore]
    SES[SessionManager / SessionStorage]
    LR[Long-running checkpoints / SchedulerLedger / DLQ]
    EV[Evidence plane · usage · observability]
    CATP[FederatedCapabilityCatalog + UCA]
    SCAF[intergrax.scaffold + init_scenario_implementation]
  end
  SEF[SEF-X EEA — after GATE-SEF-CONSUME]
  ADM[Scenario programme admission owner (OD-SFX-13) + PROD-Q]
  ST-->API; IDE-->CLI; IDE-.->MCP
  API & CLI & MCP --> SVC
  SVC --> CON & AG & RES & BP & INT & HO & QUAL & REP & CATS
  PERS --> CW & SES & LR
  AG --> EXE
  RES --> CATP
  HO --> SCAF
  HO -.-> SEF
  REP --> EV
  QUAL -->|admission dossier only| ADM
```

### 7.3 Subsystem ownership and no-duplicate review

As in R0 §7.3. ▲ Changed rows:
- Subsystem 8 now has the two engineering modes.
- Subsystem 12, the catalog, is a **sidecar document owned by Tier-2 SFX**, persisted as a `WorkArtifactVersion`. It is modelled on `AgentCatalogEntry` and is "metadata only, never execution truth". It is not a registry and creates no semantic authority.

### 7.4 Typed interface inventory (all PROPOSED; `sfx.<name>.v1`)

**Contracts:**
- SFX-CON-01 `BusinessProblemStatement`
- 02 `DiscoveryRecord`
- 03 `BusinessScenarioSpecification`
- 04 `CapabilityResolutionReport`
- 05 `ScenarioBlueprint`
- 06 `IntegrationSetupPlan`
- 07 `ScenarioTestPlan`
- 08 `EngineeringHandoffRequest` (`mode ∈ {DETERMINISTIC_SCAFFOLD, AGENTIC_ENGINEERING}`; `blueprint_digest`; `sef_intent_draft?`)
- 09 `ScenarioRunReport`
- 10 `ScenarioValueReport`
- 11 `ScenarioCatalogEntry` (sidecar)
- 12 `SfxSessionState`
- 13 `ScenarioApprovalRef` (wraps the decision binding)
- ▼14 `ScenarioLifecycleRecord`
- ▼15 `EngineeringJobRecord` (an SFX view that holds the SEF `job_id` by reference)
- ▼16 `ScenarioQualificationRecord`
- ▼17 `ScenarioAdmissionDossier` (an export, not an admission)

**Ports:** as in R0, plus `ScenarioPersistence` (adapters over CW, Session and LR) and `ScenarioQualifier`.

**Names avoided:** CON-X-01/02 and CON-R-06.

### 7.5 Durable, versioned lifecycle (▲ new — reuses existing owners only)

| Record | Owner (existing) | Versioning | Lifecycle / state | Tenant scope | Retention |
|---|---|---|---|---|---|
| Scenario container | `WorkItem` (one per scenario) | `revision` | `WorkItemState` OPEN→ACTIVE→COMPLETED/CANCELLED (coarse) | tenant + workspace | business class |
| Session | `SessionManager` + `SessionStorage`; `ChatSession.metadata` = {sfx_scenario_id, work_item_id} | — (conversation) | OPEN/CLOSED | tenant | MemoryControlPlane sweep (ADR-PDC-02) |
| Session resume state (`SfxSessionState`) | `WorkArtifact` "sfx-session-state" | append-only versions | current pointer (CAS) | tenant + workspace | business |
| Spec, blueprint, test plan, integration plan | `WorkArtifact` + `WorkArtifactVersion` + `ArtifactContentStore` | immutable versions, `publish_version` CAS, content digest = canonical digest (§7.7) | via lifecycle record | tenant + workspace | business, never deleted while referenced |
| SFX lifecycle (DRAFT…ADMISSION_DOSSIER_READY) | `WorkArtifact` "sfx-lifecycle". Each transition is a new `WorkArtifactVersion` with `ScenarioLifecycleRecord{from,to,actor,evidence_refs}`; the current state is the current pointer | append-only | FSM §7.9; gated transitions require a decision binding | tenant + workspace | audit-grade (no delete) |
| Approval | `CollaborativeDecisionBindingRepository` (binds a WorkArtifactVersion to a `DecisionProposalRef`); runtime pauses go through `HumanDecisionPersistence` | append-only | proposed→approved/rejected | tenant | audit evidence (ADR-PDC-08: never deleted) |
| Engineering job | `Assignment` + `WorkItemExecutionLink`; scaffold job execution through long-running `TaskCheckpoint`; agentic job state **remains in SEF** `EngineeringExecutionRecord` (referenced) | checkpoint `revision` | QUEUED/RUNNING/WAITING/SUCCEEDED/FAILED/DEAD_LETTERED (mapped onto LR states) | tenant | per LR policy |
| Run facts | Evidence plane (`EvidencePersistencePort`) | append-only | — | tenant | evidence class |
| Run / ROI report | `WorkArtifactVersion` with `execution` provenance → evidence refs | immutable | — | tenant + workspace | business |
| Catalog sidecar | `WorkArtifactVersion` keyed by `ApplicationPackage.package_id` + version + package checksum | immutable | listed / deprecated / retired (inside the sidecar) | tenant + workspace | business |

**Limits.** `WorkArtifact` has no lifecycle field, and `WorkItemState` is coarse; the lifecycle-artifact pattern above covers both. There is no new store and no new state engine. The CW qualification state is REQUALIFICATION_REQUIRED (SFX-P1-03).

### 7.6 Catalog without changing `ApplicationManifest` (▲)

- **Default:** use the sidecar `sfx.scenario_catalog_entry.v1`. It is owned by Tier-2 SFX, references the package by `package_id`, version and checksum, and is persisted as a versioned CW artifact. `ApplicationManifest`, `ApplicationPackage` and `ApplicationGraphSpec` stay unchanged. `EnvironmentExtensions` must not be used, because those are policy fragments.
- **Only if** a later requirement proves that a manifest field is unavoidable, follow this path:
  1. Record the justification in an ADR.
  2. Use the COMPAT-X additive policy, with family compatibility policy plus evidence.
  3. Run the EVOL-X change-impact selection.
  4. Apply ECAP D1–D6 class C and an Architecture Reopen Record.
  5. Requalify targets.
  6. Get a separate operator OD.
- No WBS task depends on this path.

### 7.7 Parity contract Studio ↔ CLI ↔ MCP (▲ corrected)

| ID | Rule | Test |
|---|---|---|
| PC-1 Schemas | One schema set, generated from the Tier-2 pydantic contracts (`agents/scenario_factory/schemas/*.schema.json`). The Studio client and the MCP tool schemas are generated from the same source. | Schema-drift test: generated output equals the committed schemas. |
| PC-2 Validation | All doors call the same Tier-2 service functions. Doors hold no validation logic beyond transport parsing. | Error-code parity: an identical invalid input gives an identical `sfx.error` code and pointer in all doors. |
| PC-3 Permissions | One authorization path (harness principal → tenant reconciliation → PolicyEngine). The offline CLI is limited to **L0/L1 on local files**; no shared-store writes or L2 runs. Shared writes and L2 runs go only through an authenticated door (API or MCP) with the same principal model. | Permission matrix test per authority level × door. |
| PC-4 Semantics | Identical artifact identity (`artifact_id`, version, `canonical_digest`), lifecycle transitions and side effects for the same operation. | Op-equivalence test over the service operation catalog. |
| PC-5 Canonicalization | The canonical form is RFC 8785 JSON Canonicalization of the model dump; `canonical_digest = sha256(canonical form)`. YAML and pretty JSON are presentations and are never hashed. The determinism scope is the deterministic stages (validate, resolve, compose, canonicalize, scaffold). LLM stages are compared by replaying recorded turns. | Byte-identity of the canonical form across doors; replay equivalence. |
| PC-6 Versioning | API v1 = MCP tool set v1 = CLI v1, with a 1:1 operation map and one version constant. | Operation-map completeness test. |

### 7.8 Studio stack verification (OD-SFX-02 kept; ▲ consequences and missing evidence)

**Consistent with existing patterns (VERIFIED):**
- Tier-3 hosts build FastAPI through `create_app(ApiConfig)` plus `mount_*_routes`, with optional MCP coupling (`couple_harness_host_fastapi_with_mcp`).
- DX-MAINT-04 places product UI in Tier-3 hosts.
- §8.6 "UI must not define the architecture" holds because the API is contract-first from the Tier-2 schemas (PC-1).
- Boundary scans read only `*.py`.
- App discovery reads `applications/<pkg>/package.json` at the app root only (`check_application_package.py:90`).

**Consequences and required work:**

| ID | Consequence | Task |
|---|---|---|
| S-1 | No Node in CI. Add a path-filtered workflow `scenario-studio-web.yml` (lint, typecheck, unit, build, e2e, npm audit). Workflow edits are governed paths. | SFX-P10-04; OD-SFX-05 |
| S-2 | No browser auth. JWT bearer exists in `fastapi_core/auth/providers/jwt` but is unwired; there are no cookies or OIDC. Studio v1 uses a JWT bearer with an explicit CORS origin list (no `*`) and no cookie session. | SFX-P10-01; ADR-SFX-003 |
| S-3 | No `StaticFiles` or SPA fallback. Use a dev proxy (Vite) and a production build mount inside the Tier-3 host (no `fastapi_core` change). | SFX-P10-07 |
| S-4 | OpenAPI is disabled in PROD. Export the schema at build time for client generation. | SFX-P10-01 |
| S-5 | `.gitignore` lacks `node_modules/`. `package-lock.json` is a tracked JSON that the IIAC G0 inventory will pick up. | SFX-P5-06; OD-SFX-05 |
| S-6 | Name confusion: the app-root `package.json` is an `ApplicationPackage`, while `web/package.json` is npm. Keep the npm manifest strictly under `web/`; add a test that `web/` is excluded from package discovery. | SFX-P5-06 |
| S-7 | A root `pyproject.toml` workspace member entry is needed for the new app. | SFX-P5-06 |

**Missing evidence:**
- **ME-1:** whether a new `applications/<x>` after freeze is Class A (likely, "new domain module attached via contracts"). Confirm in ADR-SFX-001 review.
- **ME-2:** whether adopting a JS toolchain needs its own ADR. ADR-SFX-003 covers it by default.
- **ME-3:** the Phase K definition referenced by DX-MAINT-04.
- **ME-4:** whether shipped-product manifest checks require a catalog entry for the new app.
- **ME-5:** Windows support of the Node toolchain and Playwright under the original-environment rule.

### 7.9 Lifecycle state machine (▲ corrected: SFX stops before production)

```mermaid
stateDiagram-v2
  [*] --> DRAFT
  DRAFT --> DISCOVERED
  DISCOVERED --> SPECIFIED
  SPECIFIED --> BLUEPRINTED
  BLUEPRINTED --> APPROVED: decision binding (human)
  APPROVED --> BUILDING: handoff (mode)
  BUILDING --> BUILT
  BUILDING --> APPROVED: job FAILED (no auto-retry beyond policy)
  BUILT --> SANDBOX_TESTED: SFX qualification L2
  SANDBOX_TESTED --> SANDBOX_CERTIFIED: independent check (not implementer)
  SANDBOX_CERTIFIED --> ADMISSION_DOSSIER_READY: dossier export
  SANDBOX_TESTED --> BLUEPRINTED: improve (new version)
  ADMISSION_DOSSIER_READY --> [*]: outside SFX — admission owner (OD-SFX-13) + PROD-Q
```

**Mapping to `scenario_lifecycle.py`:**
- DRAFT..BLUEPRINTED → DESIGN
- APPROVED → ACCEPTED_FOR_IMPLEMENTATION
- BUILT → IMPLEMENTATION_INITIALIZED
- SANDBOX_TESTED → EXECUTABLE
- SANDBOX_CERTIFIED → VERIFIED

SFX treats the markdown lifecycle value as **informational**. The authoritative state is the CW lifecycle record, because frontmatter is hand-editable (§14.2).

### 7.10 Authority levels

L0 read-only · L1 proposal-only · L2 sandbox-executable (LAB, fixture adapters, budgets, egress deny) · L3 production. **L3 is never granted by SFX.**

### 7.11 ADRs (P2, all Proposed)

| ADR | Topic |
|---|---|
| ADR-SFX-001 | Placement & ownership (Option B). Includes the ECAP §16 applicability question (C-ECAP-01). |
| ADR-SFX-002 | Contracts, canonicalization, workspace location (`applications/<slug>_scenario/`), persistence mapping §7.5 |
| ADR-SFX-003 | Studio stack, auth, hosting (§7.8) |
| ADR-SFX-004 | IDE toolkit + CLI + MCP parity (§7.7) |
| ADR-SFX-005 | Catalog sidecar (§7.6) |
| ADR-SFX-006 | Authority levels, engineering modes, admission separation (§3, §14) |

---

## 8. UX architecture and generalization test

**UX (unchanged from R0):**
- Lifecycle: Describe → Discover → Understand → Design → Configure → Approve → Build → Test → Run → Present → Improve.
- Business journey runs in the Studio; the engineer journey runs in the IDE via `/sfx-session` and CLI/MCP.
- Interaction: progressive disclosure, dynamic forms from JSON Schema, conversational refinement, a form-edited visual canvas, a config inspector, a version diff, an approval inbox, interruption recovery, and authority badges.
- Usability: three rounds of five participants. TTFS, task success and SUS targets are PROPOSED (OD-SFX-07).

**▲ Generalization test (factory, not scenario).** The factory must work on problems it was never tuned on.
- **GEN set:** at least five independent business problems from different domains, plus REF-1..3.
- The GEN set is authored by an **independent session** (not the implementer or architect). It is sealed by digest and committed as a hash only, before W3 starts. The content is revealed only at SFX-P12-04.
- **No tuning on GEN:** after reveal, any change to factory code requires re-running the full GEN set and REF-1..3 at the new SHA.

**Metrics per scenario (definitions fixed in P2; targets per OD-SFX-07):**

| Metric | Definition |
|---|---|
| G-1 Reuse ratio | Share of blueprint capability bindings that are REUSE_AS_IS or REUSE_WITH_CONFIGURATION |
| G-2 Manual interventions | Count of human edits outside the designed approval and answer steps, logged by the lifecycle record actor |
| G-3 Generation success | First-pass rates of: spec valid, blueprint resolvable, build conformant, sandbox tests pass |
| G-4 Rework | Number of improve loops, plus the share of generated lines changed before SANDBOX_CERTIFIED (diff vs generated digest) |
| G-5 | Time and cost per stage |

Results are recorded per scenario and as the aggregate distribution. REF results and GEN results are reported **separately**.

---

## 9. Requirements traceability (R0 rows kept; ▼ new rows)

| REQ | Requirement | Tasks |
|---|---|---|
| REQ-SFX-041 | Two engineering modes | SFX-P8-01, P8-03 |
| REQ-SFX-042 | Certifiable handoff HO-1..7 | SFX-P8-03, P12-01..03 |
| REQ-SFX-043 | Durable versioned lifecycle §7.5 | SFX-P5-05, P6-03, P8-02 |
| REQ-SFX-044 | No frozen-manifest change | SFX-P11-03; gate in P13-02 |
| REQ-SFX-045 | Parity PC-1..6 | SFX-P5-04, P10-03 |
| REQ-SFX-046 | Generalization test | SFX-P3-02 (seal), P12-04 |
| REQ-SFX-047 | Generated-code SAST | SFX-P9-03 |
| REQ-SFX-048 | Dependency scanning and allowlist | SFX-P9-03, P10-04 |
| REQ-SFX-049 | Workspace isolation | SFX-P9-04 |
| REQ-SFX-050 | Network restriction | SFX-P9-04; C-SEF-02 |
| REQ-SFX-051 | Safe tools / no self-escalation | SFX-P9-02, P10-02 |
| REQ-SFX-052 | Product certification separated from scenario admission | SFX-P11-04, P13-03 |
| REQ-SFX-053 | Safe publication procedure | §15.3, every wave |
| REQ-SFX-054 | Semantic gate assessment | SFX-P2-01 (ADR-SFX-006), P11-04 |

The R0 rows REQ-SFX-001..040 are unchanged. The full matrix lives in `SFX_REQUIREMENTS_TRACEABILITY.json` (created in W0).

---

## 10. Master WBS (▲ split into auditable units)

**Columns:** ID | Description | Files | Deps | Tests | Evidence | Acceptance | Owner · Risk · Par.

**Per-unit rules:**
- One writer per file set.
- The evidence log goes to `.tmp/proof/sfx/<unit>/`.
- One `uv run pytest` per verification step.
- Every unit names its ticket at wave planning.

### Pre-freeze (W0–W2; classification in §15.1)

| ID | Description | Files | Deps | Tests | Evidence | Acceptance | Owner·Risk·Par |
|---|---|---|---|---|---|---|---|
| SFX-P0-01 | Ticket T-IGX-037-A, plan, digest, OD-SFX-01..04 records | `governance/tickets/T-IGX-037-A.md`, `…/plans/PLAN_T-IGX-037-A.md`, `…/scenario_factory_x/operator_decisions/OD-SFX-0{1..4}.md` | — | `verify_plan_approval` | ticket | Approved line by operator/delegate | architect · L · — |
| SFX-P0-02 | Roadmap A1–A10 | roadmap | P0-01 | `roadmap_replay_x` | log | no new failures vs `eadb1c21` | architect · M · — |
| SFX-P0-03 | SSOT plan + capability pair + dossier skeleton | `governance/tickets/plans/SFX-MASTER-PLAN.md`; `docs/project/capabilities/{architecture,plan}/SCENARIO_FACTORY.md`; `capabilities/README.md`; `qualification/scenario_factory_x/{SFX_INDEX,SFX_WBS,SFX_GAP_REGISTER,SFX_SHA_LADDER}.md`, `SFX_REQUIREMENTS_TRACEABILITY.json` | P0-01 | `tests/unit/governance` | log | links valid | architect · L · ∥P0-02 |
| SFX-P1-01 | Competitive evidence matrix | `…/satellites/scenario_factory/SFX_COMPETITIVE_EVIDENCE_MATRIX.md` | W0 | citation check (each cell has URL+date or UNKNOWN) | matrix | 100% cited/UNKNOWN | research ×3 lanes · M |
| SFX-P1-02 | IDE + MCP client facts | `…/SFX_IDE_TOOLING_FACTS.md` | W0 | — | cited | resolves GAP-07/ME | research · L · ∥ |
| SFX-P1-03 | Inventory re-pin incl. CW qualification status | `…/SFX_CAPABILITY_INVENTORY.json` | W0 | JSON schema check | inventory | every CAP classified | architect · M · ∥ |
| SFX-P2-01 | ADR-SFX-001..006 + ownership review (ToolRegistry, ECAP §16) | `docs/project/technical/adr/entries/<date>/ADR-SFX-00{1..6}.md` | P1 | `check_harness_adr.py` | ADRs | independent review (reviewer session) | architect · M |
| SFX-P2-02 | Contract drafts as JSON Schema (docs) + canonicalization spec + metric definitions G-1..5 | `…/scenario_factory/design/schemas/*.schema.json`, `SFX_CANONICALIZATION.md`, `SFX_METRICS.md` | P2-01 | JSON schema lint | files | review | architect · M · ∥P3 |
| SFX-P3-01 | UX journeys, IA, static prototype (Artifact, outside the repo product paths), usability protocol, round 0 | `…/SFX_UX_DESIGN.md` | P2-01 | round 0 (5 users) | report | TTFS baseline | ux · M · ∥P2-02 |
| SFX-P3-02 | GEN set authored and sealed by an independent session | `…/SFX_GEN_SET_SEAL.md` (digest only) | P2-02 | digest verify | seal record | sealed before W3 | independent · L |
| SFX-P4-A01 | IDE v0 declarative: rule `intergrax-scenario-factory` (`loading: auto`), skills `sfx-session`, `sfx-discover`, `sfx-specify`, `sfx-resolve`, `sfx-blueprint`, `sfx-integration-setup`, `sfx-test-plan`, `sfx-present`; L1 output to `.tmp/sfx/<slug>/` against the P2-02 schemas | `agent-env/rules/intergrax-scenario-factory.md`, `agent-env/skills/sfx-*/SKILL.md` + regenerated adapters `.claude/skills/*`, `.cursor/{rules,skills}/*` (no `CLAUDE.md` block change because the rule is `auto`) | P2-02 | `test_agent_env_sync.py`, `check_cursor_token_setup.py` | logs | `--check` clean; operator walkthrough on REF-1 text | implementer · L |
| SFX-P4-A02 | Agent persona briefs (declarative, consumed by skills as subagent briefs; not synced) | `agent-env/agents/sfx-{discovery-analyst,scenario-architect,capability-resolver,integration-advisor,test-designer,critic}.md` | P4-A01 | `--check` unaffected (directory not globbed) | log | personas read-only except architect drafts | implementer · L · ∥ |

### Gate: GATE-SFX-ENTRY

All of the following must hold:
- ARCH-FREEZE is CLOSED.
- ADR-SFX-001..006 are Accepted (OD-SFX-06).
- OD-SFX-05/09/14 are signed. OD-SFX-13 is not an entry condition; it gates admission only.
- The inventory is re-pinned at `FREEZE_CERTIFICATION_SHA`.
- The GEN set is sealed.

### Post-freeze waves

Every wave ends with **REBIND-Wn**: ECAP A1–A8 delta audit, D1–D6 classification (default C), the IIAC ACT-N11 audited re-bind, and an independent reviewer session.

**W3 — foundations (P5)**

| ID | Description | Files | Deps | Tests | Evidence | Acceptance | Owner·Risk·Par |
|---|---|---|---|---|---|---|---|
| SFX-P4-B01 | Extend `sync_agent_env.py` to generate agent definitions (`.claude/agents/`, Cursor per P1-02) | `scripts/governance/sync_agent_env.py`, `tests/unit/governance/test_agent_env_sync.py` | GATE | sync tests | log | `--check` clean | implementer · L · ∥ |
| SFX-P4-B02 | role_registry `subagent_role_map` entries for sfx agents + pinned test update | `governance/agent-control-plane/role_registry.yaml`, `tests/unit/governance/test_agent_role_routing.py` | P4-B01 | role routing + DCP registry tests | log | 7 roles unchanged | implementer · L |
| SFX-P5-01 | Tier-2 package via `intergrax.scaffold new-agent` | `agents/scenario_factory/{__init__.py,pyproject.toml,README.md}` | GATE | `test_tier_dependency_boundaries` | log | boundaries green | implementer · L |
| SFX-P5-02 | Contracts CON-01..07 | `agents/scenario_factory/contracts/{problem,discovery,specification,resolution,blueprint,integration,testing}.py` | P5-01 | `tests/unit/agents/scenario_factory/contracts/` round-trip | log | 100% schema round-trip | implementer · M |
| SFX-P5-03 | Contracts CON-08..17 | `…/contracts/{handoff,reporting,catalog,session,lifecycle,approval,job,qualification,admission}.py` | P5-01 | same | log | same | implementer · M · ∥P5-02 |
| SFX-P5-04 | Canonicalizer (RFC 8785) + schema export + drift test | `…/canonical.py`, `…/schemas/*.schema.json`, `tests/…/test_schema_drift.py`, `test_canonical.py` | P5-02/03 | canonical vectors | log | deterministic digests | implementer · M |
| SFX-P5-05 | Persistence adapters over CW, Session and LR (§7.5) incl. lifecycle-record CAS | `…/persistence/{work_items,artifacts,lifecycle,decisions,jobs}.py` | P5-04 | in-memory + SQLite backends; tenant-crossover deny | log | append-only + CAS + tenant isolation | implementer · H |
| SFX-P5-06 | Tier-3 host via `new-application`; root workspace member; `.gitignore` `node_modules/`; `web/` excluded from package discovery | `applications/scenario_studio_application/{host/factory.py,serving/fastapi_router.py,manifest.py,package.json,pyproject.toml}`, root `pyproject.toml`, `.gitignore` | P5-01 | host build + `check_application_package` | log | host boots LAB | implementer · M · ∥P5-02 |
| SFX-P5-07 | Service layer + CLI v0 (validate/show/canonicalize; offline L0/L1) | `agents/scenario_factory/service/*.py`, `applications/scenario_studio_application/cli/__main__.py` | P5-04, P5-06 | CLI tests | log | PC-2 for CLI | implementer · M |

**W4 — discovery, design, engineering handoff (P6–P8)**

| ID | Description | Files | Deps | Tests | Evidence | Acceptance | Owner·Risk·Par |
|---|---|---|---|---|---|---|---|
| SFX-P6-01 | Discovery agent (AgentContract **lifecycle STAGING explicit**), LAB via NexusLoop | `agents/scenario_factory/discovery/{agent.py,steps/,prompts/}` | W3 | scripted-interview replay fixtures `tests/fixtures/scenario_factory/discovery/` | eval report | completeness ≥ OD-SFX-07 target | implementer · H · lane A |
| SFX-P6-02 | Spec author (LLM) + deterministic validator | `…/specification/` | P6-01 | validator unit + replay | log | invalid specs rejected with pointer | implementer · M · lane A |
| SFX-P6-03 | Session resume (SessionManager + `SfxSessionState` artifact) | `…/service/session.py` | P5-05 | interruption/resume tests | log | resume in either door | implementer · M · lane A |
| SFX-P7-01 | Capability resolver (catalog + UCA gap classes), deterministic | `…/resolution/` | W3 | golden tests | log | REF-1 golden | implementer · M · lane B |
| SFX-P7-02 | Blueprint + profile composer (L2 profile: LAB, fixture adapters, egress deny) | `…/blueprint/` | P7-01 | golden blueprints | log | deterministic digest | implementer · M · lane B |
| SFX-P7-03 | Integration setup assistant (secret refs only) | `…/integration/` | P7-02 | no-secret-value scan | log | 0 secret values | implementer · M · lane B |
| SFX-P7-04 | Blueprint versioning + semantic diff | `…/blueprint/diff.py` | P7-02 | diff tests | log | stable diff | implementer · L · lane B |
| SFX-P8-01 | `EngineeringHandoffPort` + `DETERMINISTIC_SCAFFOLD` adapter | `…/handoff/{port.py,scaffold_adapter.py}` | P7-02 | determinism (same blueprint → same tree digest) + `scenario_architecture_conformance` | log | conformant skeleton | implementer · M |
| SFX-P8-02 | Engineering job lifecycle (Assignment, ExecutionLink, LR checkpoint, DLQ) | `…/handoff/jobs.py` | P8-01, P5-05 | retry/DLQ/idempotency | log | no duplicate job effects | implementer · M |
| SFX-P8-03 | `AGENTIC_ENGINEERING` adapter: intent draft, human submit, SEF record tracking, bundle verification (HO-1..7); flag-gated | `…/handoff/sef_adapter.py` | P8-02 + **GATE-SEF-CONSUME** | contract tests vs SEF schemas; HO tests | log | HO-1..7 green on SEF fixtures | implementer · H |

**W5 — qualification and front doors (P9, P10)**

| ID | Description | Files | Deps | Tests | Evidence | Acceptance | Owner·Risk·Par |
|---|---|---|---|---|---|---|---|
| SFX-P9-01 | Test planner + runner adapter (EvalRunner, proof runner, conformance) | `…/testing/` | W4 | unit | log | REF-1 plan runs L2 | tester+impl · M · lane C |
| SFX-P9-02 | Generated-code no-self-escalation gates (AST; reuse `agent_lifecycle_bypass_ast.py`) | `tests/qualification/scenario_factory_x/generated_code/test_no_self_escalation.py`, `…/qualification/escalation_rules.py` | P8-01 | mutation tests (each forbidden pattern injected → rejected) | log | 100% patterns rejected | tester · H · lane C |
| SFX-P9-03 | SAST + dependency gates for generated packages and web (semgrep / ruff-S / pip-audit / npm audit per OD-SFX-14); dependency allowlist | `…/qualification/supply_chain.py`, `tests/qualification/scenario_factory_x/supply_chain/` | P8-01 | seeded-vuln fixtures | log | seeded findings detected; clean REF passes | tester · M · lane C |
| SFX-P9-04 | Workspace + network qualification: scaffold writes confined; L2 egress deny; agentic path asserts SEF lease/egress evidence | `tests/qualification/scenario_factory_x/isolation/` | P8-01/03 | path-escape, egress, lease-absent tests | log | all fail closed | tester · H · lane C |
| SFX-P9-05 | Adversarial suite (injection in discovery/uploads/RAG, tenant crossover, secret leakage, budget, retry) | `tests/qualification/scenario_factory_x/adversarial/` | P9-01 | suite | log | all fail closed | tester · H |
| SFX-P10-01 | Studio API v1 (JWT bearer, explicit CORS, OpenAPI export) | `applications/scenario_studio_application/serving/{api_v1.py,openapi_export.py}` | W4 | API + auth tests | log | PC-2/3 for API | implementer · M · lane D |
| SFX-P10-02 | MCP surface, explicit allowlist, no-widening test | `…/mcp/server.py` | P10-01 | MCP tests (streamable HTTP) | log | allowlist == SFX op set | implementer · M · lane D |
| SFX-P10-03 | Parity suite PC-1..6 | `tests/qualification/scenario_factory_x/parity/` | P10-01/02, P5-07 | parity | log | 100% PC tests | tester · M |
| SFX-P10-04 | Web foundation (Vite+React+TS, lint, typecheck, design system, generated client) + CI workflow | `…/web/{package.json,vite.config.ts,src/…}`, `.github/workflows/scenario-studio-web.yml` | P10-01, OD-SFX-05 | unit + build + npm audit | CI run | green, path-filtered | implementer(fe) · H · lane E |
| SFX-P10-05 | Screens Describe→Configure | `…/web/src/features/{describe,discover,understand,design,configure}/` | P10-04 | unit | log | flows on REF-1 | fe · M · lane E |
| SFX-P10-06 | Screens Approve→Improve + approval inbox + version history + e2e (Playwright) | `…/web/src/features/{approve,build,test,run,present,improve}/`, `…/web/e2e/` | P10-05 | e2e | log | full lifecycle walkable | fe · H · lane E |
| SFX-P10-07 | SPA hosting (dev proxy + build mount in Tier-3 host) | `…/host/spa_mount.py` | P10-04 | host tests | log | no `fastapi_core` change | implementer · L |
| SFX-P10-08 | IDE toolkit v1 (skills call CLI/MCP) | `agent-env/skills/sfx-*` + adapters | P10-02, P4-B01 | sync + parity | log | IDE ≡ Studio artifacts | implementer · M |

**W6 — reporting, catalog, admission dossier (P11)**

| ID | Description | Files | Deps | Tests | Evidence | Acceptance | Owner·Risk·Par |
|---|---|---|---|---|---|---|---|
| SFX-P11-01 | Run reporter (evidence read port) | `…/reporting/run_report.py` | W5 | golden reports | log | facts traceable to evidence | implementer · M · ∥ |
| SFX-P11-02 | ROI / value model (inputs declared, no fabricated baselines) | `…/reporting/value_model.py` | P11-01 | unit | log | every number has a source ref | implementer · M |
| SFX-P11-03 | Catalog sidecar + reuse/clone | `…/catalog/` | W5 | unit + "manifest unchanged" gate | log | no `ApplicationManifest` diff | implementer · M · ∥ |
| SFX-P11-04 | Admission dossier export (§14.3) | `…/qualification/admission_dossier.py` | P9-*, P11-01 | dossier completeness | log | dossier ≠ admission | implementer · M |

**W7 — pilots (P12)**

| ID | Description | Deps | Evidence | Acceptance |
|---|---|---|---|---|
| SFX-P12-01..03 | REF-1/2/3 end to end: deterministic baseline, then `AGENTIC_ENGINEERING` | W6 + GATE-SEF-CONSUME | evidence bundles @ SHA | §12 acceptance |
| SFX-P12-04 | GEN set reveal, run, metrics G-1..5 (independent evaluator) | P12-01 | GEN report | OD-SFX-07 targets |
| SFX-P12-05 | Usability rounds 1–3 + refinement backlog burn-down | P10-06 | reports | 0 critical defects |
| SFX-P12-06 | Performance measurement §13 | P12-01..03 | metrics @ SHA | recorded with artifacts |

**W8 — closure (P13)**

| ID | Description | Acceptance |
|---|---|---|
| SFX-P13-01 | Evidence publication + SHA ladder | complete ladder |
| SFX-P13-02 | ENT-AUDIT CHANGED-SINCE + "manifest unchanged" + no-new-authority checks | 0 new violations |
| SFX-P13-03 | IIAC independent **product** certification (separate session) | `SCENARIO-FACTORY-X — CLOSED — INDEPENDENTLY AUDITED (INTERNAL AGENT CERTIFICATION) — PUBLISHED` |
| SFX-P13-04 | Operator closure OD-SFX-11 + roadmap row update | signed |

**Separation of duties per unit:**
- investigation: architect;
- implementation: implementer;
- review: reviewer, in a separate session;
- per-scenario sandbox certification: an independent evaluator session;
- product certification: independent-quality-auditor;
- acceptance: operator.

---

## 11. DAG, critical path, waves, worktrees

```mermaid
flowchart LR
  W0-->W1-->W2
  AF[ARCH-FREEZE CLOSED]-->GE[GATE-SFX-ENTRY]
  W2-->GE-->W3-->W4-->W5-->W6-->W7-->W8
  SEFG[GATE-SEF-CONSUME: ADR-SEF-001 Accepted + GATE-SEF-FULL + SEF B6]-.->P83[SFX-P8-03]
  SEFG-->W7
  RK[READINESS kit CLOSED]-.optional.->P81[SFX-P8-01 templates]
  W8-.dossiers.->ADM[Scenario admission owner + PROD-Q]
```

- **Critical path:** ARCH-FREEZE → GATE-SFX-ENTRY → W3 → W4 → W5 → W6 → **max(W6, GATE-SEF-CONSUME)** → W7 → W8. SEF-X engineering certification therefore sits on the SFX closure path. SEF-X and READINESS do not depend on SFX.
- **Lanes:** A (discovery), B (design), C (qualification), D (API/MCP), E (web). Each lane has its own worktree, e.g. `_wt/intergrax/sfx-w4-disc`, `sfx-w4-bp`, `sfx-w5-qual`, `sfx-w5-api`, `sfx-w5-web`. Lanes integrate into the wave branch, and one PR per wave goes to `development`.
- **Agent limits:** at most 3 explore and 2 implementer agents per wave.
- **SHA ladder:** as in R0, plus `GEN_SEAL_SHA`, `GATE_SEF_CONSUME_REF`, and the REBIND-Wn activation SHAs.

---

## 12. Reference scenarios

REF-1, REF-2 and REF-3 are as in R0, with these changes:
- Every REF runs **both** modes. Only the agentic run counts toward GATE-SFX-FULL.
- Every generated `AgentContract` must declare a non-PRODUCTION lifecycle (enforced by SFX-P9-02).
- All REFs stop at ADMISSION_DOSSIER_READY.
- No runtime results are claimed.
- REF-2 and REF-3 stay subject to audit (OD-SFX-08). REF-2 must not duplicate the READINESS settlement reference.

---

## 13. Performance and security qualification (▲ extended)

**Performance:** metric families M-1..M-9 as in R0, plus G-1..G-5 (§8). There are no values without run artifacts.

**Generated-code and execution security (SFX qualification scope):**

| Control | Mechanism | Task |
|---|---|---|
| Generated-code SAST | semgrep + ruff `S` rules on generated packages and `web/` (tooling per OD-SFX-14; release gate `check_harness_security_promote_gate.py` reused as a pattern) | SFX-P9-03 |
| Dependency scanning | pip-audit / npm audit; a generated package may add only allowlisted dependencies; lockfile required for `web/` | SFX-P9-03, P10-04 |
| Workspace isolation | Scaffold writes are confined to the job workspace (path-escape tests). The agentic path requires SEF `WorkspaceLease` evidence in the bundle. | SFX-P9-04 |
| Network restriction | L2 profile = fixture adapters + egress deny (`sandbox_network_egress` allowlist empty). The agentic path requires SEF egress enforcement (C-SEF-02). Provider enforcement strength is MISSING_EVIDENCE and is recorded as a limitation. | SFX-P9-04 |
| Safe tools | ToolRuntime allowlist from the blueprint; MCP explicit allowlist with a no-widening test | SFX-P7-02, P10-02 |
| No self-escalation | Generated code must not construct or mutate `ApplicationEnvironmentProfile.execution_mode`/`profile`, `policy_enforcement_mode`, `AgentGovernanceProfile`, certification records (`apply_roster_agent_governance`), `AgentContract.lifecycle_state` = PRODUCTION, exposure policies, or registries. SFX itself never holds L3 credentials. | SFX-P9-02 |
| Tenant, injection, secrets, budgets, retries, provenance | As in R0 | SFX-P5-05, P9-05 |

**Invariant SFX-INV-01:** generated code is never a certified scenario. SFX states end at SANDBOX_CERTIFIED / ADMISSION_DOSSIER_READY.

---

## 14. Certification and admission separation (▲)

### 14.1 Product certification (SFX stage)

- IIAC per-stage activation (OD-SFX-09).
- Audit by a distinct session, with exact-SHA manifests.
- ECAP per wave.
- ENT-AUDIT CHANGED-SINCE at closure.
- Scope: the factory (doors, service, contracts, persistence, handoff, gates, parity, generalization).
- **It certifies no generated scenario for production.**

### 14.2 Semantic gate assessment (replaces R0's symbol search)

- **SCENARIO-GATE is a one-time programme Go/No-Go.** The roadmap names it the "Enterprise → Scenario transition gate", and it has no per-scenario function. A missing `ScenarioAdmission` symbol is therefore expected, not a gap of that gate.
- Mechanisms that actually exist today on real execution paths:
  - STRICT+PRODUCT roster certification and revision-bound contract authority (`agent_certification_wiring.py:100-155`, `roster_agent_contract_authority.py:55-100`);
  - plugin production admission (`decision_plugin_policy.py:44-66`);
  - STRICT-gated MSE authorization;
  - the per-integration PROD-Q live slice.
- Self-promotion seams (SFX-GAP-15): `PRODUCTION_ATTACHED` is a posture flag, not an authorization; `lifecycle_state` defaults to PRODUCTION; certification records can be self-generated; the spec `VERIFIED` frontmatter is hand-editable.
- **No owner exists for an operator-signed per-scenario admission record** (SFX-GAP-16). SFX must not create one, because that would be a new semantic authority. OD-SFX-13 assigns it. Candidates are the scenario programme opened by SCENARIO-GATE, or platform Agent Runtime Governance.

### 14.3 Admission dossier

For each SANDBOX_CERTIFIED scenario, SFX exports:
- canonical artifacts plus digests;
- engineering evidence bundle;
- qualification records;
- security gate results;
- the required-integrations list for PROD-Q;
- known limitations.

The dossier is input to the admission owner. **Admission happens outside SFX** and requires: SCENARIO-GATE CLOSED, an admission decision by the OD-SFX-13 owner, PROD-Q live qualification for each integration, and a STRICT+PRODUCT host with revision-bound authority.

---

## 15. Pre-freeze classification, W0 batch and publication (▲)

### 15.1 Classification of every pre-freeze change

The applicable policy is post-freeze evolution governance A/B/C, which is ACTIVE against the earlier core-freeze baseline (`INTEGRAX_POST_FREEZE_EVOLUTION_GOVERNANCE.md:15,44`). Other relevant rules: the IIAC material paths, ACT-N11 product prefixes, and ARCH-FREEZE B1 inventory roots (`scripts/` is inventoried as a supplementary root).

| Wave/unit | Paths | Ref trailer | IIAC material | ACT-N11 | ARCH-FREEZE inventoried | Class | OD-SFX-04 |
|---|---|---|---|---|---|---|---|
| W0 P0-01..03 | `governance/tickets/**`, `docs/**` | not required (added anyway) | no | no | no | A (docs) | ✅ allowed |
| W1 P1/P2 | `docs/**`, ADR entries | no | no | no | no | A | ✅ |
| W2 P3 | docs + Artifact prototype (not in product paths) | no | no | no | no | A | ✅ |
| W2 P3-02 | seal record (docs) | no | no | no | no | A | ✅ |
| W2 P4-A01/A02 | `agent-env/**` + **generated adapters** `.claude/skills/**`, `.cursor/**` | no | no | no | no | A (dev tooling, no frozen semantics) | ✅ under C-OD-01 |
| P4-B01 | `scripts/governance/sync_agent_env.py` + test | **yes** | no | no | **yes** | A | ❌ not agent-env, so moved to W3 |
| P4-B02 | `role_registry.yaml` + pinned test | no (tests yes) | no | no | no | A | ❌ governance + test, so moved to W3 |

- **C-OD-01 (interpretation, needs confirmation at W2 ticket approval):** OD-SFX-04 says "agent-env only". The SSOT mechanism requires committing the regenerated adapters in `.claude/` and `.cursor/` in the same change. R1 reads those adapters as part of an agent-env delivery. If the operator rejects that reading, P4-A01 moves to W3.
- **R0 conflict removed:** R0 placed P4-01 (scripts) and the role_registry change pre-freeze, against OD-SFX-04. Both now default to W3.
- **OD-SFX-12 (optional):** the operator may pull P4-B01/B02 into a pre-freeze ticket as Class A tooling, but only **before** `T-B7-LOCK-FREEZE-SHA`, because `scripts/` is in the freeze inventory. This would amend OD-SFX-04 and needs an explicit signature.

### 15.2 W0 = T-IGX-037-A (first documentation batch)

The deliverables are as in R0 §15, updated to R1 (§2 wording, R1 dossier). The draft ticket and plan already exist in worktree `_wt/intergrax/sfx-p0` with state `plan_pending`. Before approval, `PLAN_T-IGX-037-A.md` must be updated to R1 (REQ range 001..054) and its digest recomputed.

- **Order:**
  1. Update the plan and digest.
  2. The operator or a delegate (separate session) writes `Approved:`.
  3. Write the docs.
  4. Run the test contract: `uv run pytest tests/qualification/roadmap_replay_x tests/unit/governance -q`, `sync_agent_env.py --check`, `igx_gov ticket`.
  5. Commit, PR, review session, operator merge.
- **Not in W0:** product paths, `scripts/`, `agent-env/`, ADRs, research findings.

### 15.3 Publication procedure (verified)

- **Workspace:** `D:\Projekty\_wt\intergrax\sfx-p0` (outside the repo, created from `origin/development` `eadb1c21`). The main checkout `D:\Projekty\intergrax` is another session's active worktree, so SFX runs no `cd`, checkout, stash, reset or write there. Read-only `git -C` queries are allowed. The plan file in `MAIN/agent-env/plans/` is untracked and is only read; its content is copied into the worktree.
- **Tracking:** `sfx-p0-registration` has upstream `origin/development`, set automatically by `worktree add -b … origin/development` (verified via `@{u}`). This keeps the pre-push ticket-ref range at `merge-base..HEAD`.
- **Push:**
  1. `git ls-remote --heads origin sfx-p0-registration` must be empty, so no existing remote branch is overwritten.
  2. `git push origin sfx-p0-registration:sfx-p0-registration`, with an explicit refspec and **no `-u`**, so the upstream does not change.
  3. Afterwards, verify that `git rev-parse --abbrev-ref @{u}` still shows `origin/development`.
  4. Never use `INTERGRAX_SKIP_TICKET_REF_GATE`.
- **PR:** `gh pr create --base development --head sfx-p0-registration`. Before asking for a merge, check that the PR `headRefOid` equals the local HEAD and that its CI has finished. The operator merges.
- **Cleanup:** after the merge, `git worktree remove` your own worktree, then `git worktree prune`. Never touch other worktrees.
- **Every later wave** uses the same procedure with its own worktree, slug and branch.

---

## 16. Blockers and operator decisions

| ID | Item | Impact | Authorization |
|---|---|---|---|
| B-SFX-01 | ARCH-FREEZE OPEN | blocks W3+ | ARCH-FREEZE closure |
| B-SFX-02 | MCP P8 / PDC-X open | transitive | their stages |
| B-SFX-03 | GATE-SEF-CONSUME not met (C-SEF-01..03) | blocks SFX-P8-03, W7 agentic, SFX closure | SEF-X track + OD-SFX-10 |
| B-SFX-04 | Research facts MISSING_EVIDENCE | scheduled in W1, not a plan defect | T-IGX-037-B |
| B-SFX-05 | OD-SFX-01..04 signatures | blocks W0 merge | operator/delegate |
| OD-SFX-05 | Node toolchain, lockfile, npm policy | W5 web | operator |
| OD-SFX-06 | ADR-SFX-001..006 Accept | GATE-SFX-ENTRY | operator |
| OD-SFX-07 | TTFS, usability, G-metric targets | P12 acceptance | operator, at P3 exit |
| OD-SFX-08 | REF-2/REF-3 inclusion | P12 scope | operator, at P2 exit |
| OD-SFX-09 | IIAC scope for SFX | W3 | operator |
| OD-SFX-10 | Cross-track edge SFX → SEF-X; SFX as SEF intake consumer | GATE-SEF-CONSUME | operator + SEF track |
| OD-SFX-11 | SFX closure | W8 | operator |
| OD-SFX-12 (optional) | Pull P4-B pre-freeze | amends OD-SFX-04 | operator |
| OD-SFX-13 | Owner of per-scenario production admission | admission only (not SFX closure) | operator |
| OD-SFX-14 | Adoption of SAST / dependency tools (semgrep, pip-audit, npm audit) in the PR pipeline | W5 | operator |

**Conflicts register** (reported; none resolved unilaterally):
- C-SEF-01..04 (§3.4).
- C-OD-01 (§15.1).
- **C-RDY-01:** SFX must not define Delivery Kit templates; REF-2 must not duplicate the READINESS reference.
- **C-AF-01:** `scripts/` changes interact with the ARCH-FREEZE inventory and SHA lock (§15.1).
- **C-IIAC-01:** every post-freeze product wave needs an ACT-N11 audited re-bind (REBIND-Wn).
- **C-ECAP-01:** ECAP §16 ("update ECAP when a new domain is added") applicability to a Tier-2 product module is decided in the ADR-SFX-001 review.

---

## 17. CORRECTION_MATRIX (R0 → R1)

| # | FINDING | ORIGINAL_PLAN_SECTION | CORRECTION | AFFECTED_TASKS | ACCEPTANCE | REMAINING_BLOCKER |
|---|---|---|---|---|---|---|
| 1 | SFX↔SEF unclear; scaffold and agentic modes mixed; closure did not require agentic engineering | R0 §3, §7.3 row 8, P8-01 | Two modes; handoff HO-1..7 with sequence; GATE-SEF-CONSUME; closure requires agentic REF + GEN on certified SEF; C-SEF-01..04 | SFX-P8-01/02/03, P12-01..04, A4/A6 | HO tests green; REF agentic evidence; GATE-SFX-FULL includes SEF B6 | B-SFX-03, OD-SFX-10 |
| 2 | P4-01/P4-02 (scripts, role_registry) conflicted with OD-SFX-04 | R0 §10 P4, §15 | Split into P4-A (declarative agent-env + generated adapters) pre-freeze and P4-B (scripts, registry, pinned test) in W3; full classification table §15.1 | SFX-P4-A01/A02, P4-B01/B02 | each unit classified; `--check` clean | C-OD-01 confirmation; optional OD-SFX-12 |
| 3 | Stack not verified against Tier-3 patterns | R0 §7.7 ADR-003, P10-02 | §7.8: verified patterns, S-1..S-7 consequences, ME-1..ME-5 missing evidence; choice kept | SFX-P5-06, P10-01/04/07 | ADR-SFX-003 covers S/ME items | OD-SFX-05, ME-1..5 |
| 4 | No durable versioned lifecycle | R0 §7.4 | §7.5 mapping onto CW / Session / LR / Evidence / DecisionBinding; lifecycle-record artifact pattern; no new store | SFX-P5-05, P6-03, P8-02 | append-only + CAS + tenant tests | CW requalification (P1-03) |
| 5 | Default EXTEND of `ApplicationManifest` | R0 §5 CAP-12, §7.3 row 12, P11-01 | Sidecar catalog entry (AgentCatalogEntry pattern) as a CW artifact; COMPAT-X / EVOL-X / ECAP path documented only as an exception | SFX-P11-03, P13-02 | "manifest unchanged" gate | none |
| 6 | Parity defined as byte-identity of YAML | R0 §7.2 | PC-1..PC-6: shared schemas, validation, permissions, semantics, RFC 8785 canonical digest, version map; offline CLI limited to L0/L1 | SFX-P5-04/07, P10-03 | parity suite 100% | none |
| 7 | Batch rows too coarse | R0 §10 | ~45 auditable units with files, deps, tests, evidence, acceptance, lanes; REBIND per wave | all W3–W8 | each unit has evidence path | none |
| 8 | No generalization test | R0 §12/§13 | GEN set sealed by an independent session; metrics G-1..G-5; no tuning on GEN | SFX-P3-02, P12-04 | GEN report vs OD-SFX-07 | OD-SFX-07 |
| 9 | Security of generated code incomplete | R0 §14 | SAST, dependency scan and allowlist, workspace and network isolation, safe tools, no-self-escalation gates; platform seams recorded and handed to their owner | SFX-P9-02/03/04, P10-02 | mutation and seeded-vuln tests | OD-SFX-14; C-SEF-02; provider egress evidence |
| 10 | Gate status judged by symbol search | R0 §1, §6, §7.6 | §14.2 semantic assessment: SCENARIO-GATE is a programme gate; actual admission mechanisms traced; GAP-15/16 | SFX-P2-01, P11-04 | ADR-SFX-006 reviewed | OD-SFX-13 |
| 11 | Publication procedure unverified | R0 §15 | §15.3 verified tracking (`@{u}` = origin/development), explicit refspec without `-u`, ls-remote check, headRefOid check, no main-checkout writes | every wave | post-push `@{u}` unchanged; PR head = local HEAD | none |
| 12 | Product certification and scenario admission conflated (PRODUCTION_ADMITTED inside SFX FSM) | R0 §7.6, §12, §14 | FSM ends at ADMISSION_DOSSIER_READY; §14.1 vs §14.3; A1/A4 wording | SFX-P11-04, P13-03, A1/A4/A6 | SFX closure text states no admission | OD-SFX-13 |

---

## Verification of this plan

- **Evidence base:** every new claim cites a path and line at `eadb1c21`, read from the clean worktree `sfx-p0`. Main-checkout content was not used, except this untracked plan file.
- **Consistency checks:**
  - No OD-SFX-01..04 change. One interpretation is flagged (C-OD-01) and one optional amendment (OD-SFX-12).
  - SFX is never a predecessor of READINESS or GATE. SFX closure follows SEF certification and is one-way.
  - No new registry, engine, store, FRZ ID or semantic authority. The catalog is a sidecar and admission stays external.
  - Every pre-freeze unit is classified as allowed. W0 is unchanged in nature (docs only).
  - Every new requirement REQ-SFX-041..054 is mapped to task IDs.

## Terminal status

`SFX_PLAN_READY_FOR_OPERATOR_APPROVAL`

Approval covers W0 execution under DevOS (ticket `Approved:` line still required) and planning authority for W1–W2. It does **not** authorize any post-freeze implementation (W3+), which remains gated by ARCH-FREEZE CLOSED, GATE-SFX-ENTRY and per-wave ticket approvals. Open items listed in §16 are scheduled decisions, not internal inconsistencies.
