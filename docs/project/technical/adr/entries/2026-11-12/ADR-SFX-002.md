# ADR-SFX-002: SFX contracts, versioning, canonicalization, scenario workspace location and persistence mapping

| Field | Value |
|-------|-------|
| **Status** | **Proposed** (T-IGX-037-C, wave W1; acceptance is OD-SFX-06, a separate operator review) |
| **Date** | 2026-11-12 |
| **Deciders** | Operator (OD-SFX-06 acceptance; open questions OQ-002-1..OQ-002-6 below) |
| **Reference SHA** | `eadb1c2116fe` (`origin/development`); every `path:line` citation below is at this SHA |
| **Related** | [`governance/tickets/T-IGX-037-C.md`](../../../../../../governance/tickets/T-IGX-037-C.md) · [`ADR-SEF-001`](../2026-11-05/ADR-SEF-001.md) · [`governance/tickets/plans/ASF-SEF-MASTER-PLAN.md`](../../../../../../governance/tickets/plans/ASF-SEF-MASTER-PLAN.md) (Deliverable 6) · [`COMPAT_X_CONTRACT_SCHEMA_EVOLUTION.md`](../../../../capabilities/architecture/COMPAT_X_CONTRACT_SCHEMA_EVOLUTION.md) · [`ADR_PDC_02`](../../../../maintainers/architecture/ADR_PDC_02_CONTEXT_TRIM_FIDELITY_AND_MEMORY_RETENTION.md) · [`ADR_PDC_07`](../../../../maintainers/architecture/ADR_PDC_07_BACKGROUND_RECURRENCE_AND_DLQ.md) · [`ADR_PDC_08`](../../../../maintainers/architecture/ADR_PDC_08_ARTIFACT_LIFECYCLE_TAXONOMY.md) · design drafts: [`schemas/`](../../../../maintainers/qualification/scenario_factory_x/design/schemas/), [`SFX_CANONICALIZATION.md`](../../../../maintainers/qualification/scenario_factory_x/design/SFX_CANONICALIZATION.md), [`SFX_METRICS.md`](../../../../maintainers/qualification/scenario_factory_x/design/SFX_METRICS.md) |

## Context

SCENARIO-FACTORY-X (SFX) turns a business problem into a discoverable, configurable, executable, testable and demonstrable scenario, using only existing platform mechanisms (master plan R1, Option B). Three doors (Scenario Studio, CLI, MCP) must produce identical artifacts (parity contract PC-1..PC-6, master plan R1 §7.7). That requires:

- one typed contract set with a stable version identity;
- one canonical byte form and digest per record, so that every door, the approval binding and the SEF-X handoff point at the same bytes;
- one place in the repository where generated scenarios live;
- durable, versioned persistence that reuses existing owners. SFX may not add a store, a state engine or an approval authority.

Facts at the reference SHA that constrain the design:

- `WorkArtifact` has no kind or lifecycle field: its fields are identity, scope, `current_version_id` and `revision` (`intergrax/contracts/collaborative_work.py:1238`, `:1249`, `:1250`). `WorkItemState` is coarse: OPEN, ACTIVE, COMPLETED, CANCELLED (`intergrax/contracts/collaborative_work.py:759`).
- `ArtifactPublicationRepository.publish_version` appends a version and advances the CAS pointer (`intergrax/collaborative_work/repository.py:1338`); the command carries `expected_revision` (`intergrax/collaborative_work/repository.py:1211`).
- `ArtifactContentStore.put` hashes the stored body (`intergrax/collaborative_work/content_storage.py:105`, `:185`), and the digest convention is `sha256:<64 lowercase hex>` (`intergrax/contracts/validation.py:13`, `:24`).
- The platform has no RFC 8785 implementation. Existing fingerprints use `json.dumps(sort_keys=True, separators=(",", ":"))` (for example `intergrax/collaborative_work/repository.py:216`, `:1409`).
- The scaffold always appends `_application` to the package name (`intergrax/scaffold/application_names.py:12`, `:18`).
- The proof scenario tree is `platform_proofs/scenarios/` (`scripts/proof/create_scenario_proof.py:18`), and `scenario_architecture_conformance` validates only that tree (`scripts/proof/scenario_architecture_conformance.py:12`, `:632`).

### Alternatives considered

| Variant | Summary | Disposition |
|---------|---------|-------------|
| **A. New SFX store and state engine** | Dedicated tables for scenarios, lifecycle and approvals | Rejected: a second store and a second state engine (master plan R1 Context; ADR-PDC-08 "no global storage engine", `ADR_PDC_08_ARTIFACT_LIFECYCLE_TAXONOMY.md:19`) |
| **B. Extend `WorkArtifact` / `WorkItemState` with SFX lifecycle fields** | Add a lifecycle enum to Collaborative Work | Rejected: Tier-0 product-path change after freeze; leaks a Tier-2 product lifecycle into a domain owner |
| **C. Reuse `ApprovalStorePort` for SFX approvals** | Store blueprint approvals as `ApprovalRequest` | Rejected, see D8 |
| **D. Reuse existing owners; SFX lifecycle as an append-only artifact stream (selected)** | CW artifacts + versions + CAS pointer, decision bindings, Session, long-running checkpoints, evidence plane | Selected |
| **E. Generated scenarios under `platform_proofs/scenarios/`** | Reuse the proof tree and its conformance tool unchanged | Rejected as the default: proof scenarios are CON-R-06 `ScenarioSpec_platform` (`governance/tickets/plans/ASF-SEF-MASTER-PLAN.md:360`), and SEF-X did not authorize a proof-only factory under `platform_proofs/` (ADR-SEF-001, Option 4) |

## Decision

### D1. Contract inventory (all PROPOSED; Tier-2 owner `agents/scenario_factory/`)

The design drafts are in `docs/project/maintainers/qualification/scenario_factory_x/design/schemas/sfx.<name>.v1.schema.json`. At SFX-P5 they are replaced by schemas generated from the Tier-2 pydantic models (PC-1).

| ID | Contract | `$id` | Persisted as | Notes |
|----|----------|-------|--------------|-------|
| SFX-CON-01 | BusinessProblemStatement | `sfx.business_problem_statement.v1` | CW artifact version | Free text is `*_untrusted` |
| SFX-CON-02 | DiscoveryRecord | `sfx.discovery_record.v1` | CW artifact version | LLM turns recorded for replay |
| SFX-CON-03 | BusinessScenarioSpecification | `sfx.business_scenario_specification.v1` | CW artifact version | Not `ScenarioSpec` (CON-R-06) |
| SFX-CON-04 | CapabilityResolutionReport | `sfx.capability_resolution_report.v1` | CW artifact version | Closed-world (CON-R-07) |
| SFX-CON-05 | ScenarioBlueprint | `sfx.scenario_blueprint.v1` | CW artifact version | Its digest is `D_bp`; not `ScenarioBuildPlan` (CON-X-02) |
| SFX-CON-06 | IntegrationSetupPlan | `sfx.integration_setup_plan.v1` | CW artifact version | `secret_ref` only |
| SFX-CON-07 | ScenarioTestPlan | `sfx.scenario_test_plan.v1` | CW artifact version | Targets reference OD-SFX-07 |
| SFX-CON-08 | EngineeringHandoffRequest | `sfx.engineering_handoff_request.v1` | CW artifact version | `mode`, `blueprint_digest`, `sef_intent_draft` (CON-N-01 fields) |
| SFX-CON-09 | ScenarioRunReport | `sfx.scenario_run_report.v1` | CW artifact version with `execution` provenance | References evidence; never copies it |
| SFX-CON-10 | ScenarioValueReport | `sfx.scenario_value_report.v1` | CW artifact version | Integer minor units |
| SFX-CON-11 | ScenarioCatalogEntry | `sfx.scenario_catalog_entry.v1` | CW artifact version (sidecar) | `ApplicationManifest` unchanged (ADR-SFX-005) |
| SFX-CON-12 | SfxSessionState | `sfx.sfx_session_state.v1` | CW artifact "sfx-session-state" | Resume across doors |
| SFX-CON-13 | ScenarioApprovalRef | `sfx.scenario_approval_ref.v1` | View of a `CollaborativeDecisionBinding` | Wraps the binding; no approval authority |
| SFX-CON-14 | ScenarioLifecycleRecord | `sfx.scenario_lifecycle_record.v1` | CW artifact "sfx-lifecycle" | Append-only, FSM D7 |
| SFX-CON-15 | EngineeringJobRecord | `sfx.engineering_job_record.v1` | CW artifact version + `Assignment` + execution link | SEF `job_id` by reference (HO-3); not `FactoryExecutionRecord` (CON-X-01) |
| SFX-CON-16 | ScenarioQualificationRecord | `sfx.scenario_qualification_record.v1` | CW artifact version | `certifies_production: false` |
| SFX-CON-17 | ScenarioAdmissionDossier | `sfx.scenario_admission_dossier.v1` | CW artifact version | An export; `is_admission: false` |

**Names avoided.** `FactoryExecutionRecord` (CON-X-01, `governance/tickets/plans/ASF-SEF-MASTER-PLAN.md:395`), `ScenarioBuildPlan` (CON-X-02, `:396`), and any `ScenarioSpec*` name, which would collide with CON-R-06 `ScenarioSpec_platform` (`:360`). SFX consumes the SEF contracts CON-N-01 `EngineeringIntent` (`:381`), CON-N-02 `EngineeringExecutionRecord` (`:382`), CON-N-04 `DeliveryManifest` (`:384`), CON-E-01 `EngineeringSpecification` (`:369`), CON-E-04 `VerificationRecord` (`:372`) and CON-E-05 `EngineeringEvidenceBundle` (`:373`) by reference only. It never redefines them.

**Common envelope.** Every persisted SFX record carries `schema_version` (const `sfx.<name>.v1`), `tenant_id`, `workspace_id`, `work_item_id`, `scenario_id`, `artifact_id` (equal to the CW `work_artifact_id`), `version` (integer, at least 1), `canonical_digest`, `created_at` and `created_by` (`principal_id` + `actor_kind`). Secrets appear only as `secret_ref` strings. No contract has a secret value field, and configuration keys that name secrets are rejected by the schema.

### D2. Versioning and evolution (COMPAT-X style, SFX family policy SFX-COMPAT-01)

- **Identity:** `sfx.<snake_name>.v<major>`. The identity is the JSON Schema `$id` and the `schema_version` const in each record. One version constant covers API v1 = MCP tool set v1 = CLI v1 (PC-6).
- **Owner:** SFX (Tier-2) owns version identity for its family. COMPAT-X defines how evolution is classified (`docs/project/capabilities/architecture/COMPAT_X_CONTRACT_SCHEMA_EVOLUTION.md:37`).
- **Breaking change** (a removed or renamed field, changed semantics, a narrowed or widened enum, a changed canonicalization rule): new identity `v2` (COMPAT-X rule 2, `:38`).
- **Unknown impact:** fail closed until it is classified (rule 3, `:39`).
- **Additive optional field within v1:** allowed only under this family policy and with evidence (rule 5, `:41`): (a) the schema-drift test passes; (b) a reader test loads every committed v1 fixture; (c) a digest-stability test verifies stored v1 records against their stored bytes. Additive changes never re-serialize existing records.
- **Old versions:** a stored version is never upgraded in place. Migration publishes a new artifact version under the new identity that references its predecessor. Old-version acceptance is only through an explicit reader (rule 4, `:40`).
- **Deprecation:** follows COMPAT-X R5 (`:191`): one semantic owner, one canonical replacement or `NOT_APPLICABLE`, and an explicit exit condition.

### D3. Canonicalization and digest

- Canonical form: RFC 8785 JSON Canonicalization Scheme (JCS) of the pydantic `model_dump(mode="json")` of the record, **without** the `canonical_digest` field.
- `canonical_digest = "sha256:" + hex(sha256(JCS(model_dump without canonical_digest)))`, in the platform format (`intergrax/contracts/validation.py:13`, `:24`).
- The CW content body of each SFX artifact version is exactly those JCS bytes. `ArtifactContentStore.put` computes the digest of the body (`intergrax/collaborative_work/content_storage.py:185`), so `ArtifactContentRef.integrity_digest` (`intergrax/contracts/collaborative_work.py:1221`) equals `canonical_digest`. The field itself is a derived projection: it is recomputed and compared on every read and ingest, and a mismatch fails closed.
- YAML and pretty JSON are presentations. They are never hashed.
- SFX digests are a separate family from the platform `json.dumps(sort_keys=True)` fingerprints (`intergrax/collaborative_work/repository.py:216`). The two are never compared with each other.
- The rules (CF-1..CF-8), the determinism scope and the parity tests are in `SFX_CANONICALIZATION.md`.

### D4. Scenario workspace location

- **Decision (as planned):** a generated scenario lives in a Tier-3 package under `applications/`, produced by the existing scaffold (C-SEF-04: no new kit templates). The scenario slug follows the proof slug rule `^[a-z][a-z0-9_]*$` (`scripts/proof/create_scenario_proof.py:19`).
- **Verified constraint:** the plan names the path `applications/<slug>_scenario/`. The scaffold, however, always appends `_application` (`intergrax/scaffold/application_names.py:18`). Its input `<slug>_scenario` therefore yields `applications/<slug>_scenario_application/`, with the short id `<slug>_scenario` (`intergrax/scaffold/application_names.py:23`). Producing `applications/<slug>_scenario/` exactly would need a Tier-0 scaffold change after the freeze. **Proposed resolution:** accept the scaffold output `applications/<slug>_scenario_application/` and keep `<slug>_scenario` as the logical id. This is **OQ-002-1**. The schema drafts accept both forms until it is decided.
- The generated package is discoverable as an `ApplicationPackage` at the app root (`scripts/maintenance/check_application_package.py:90`).
- **Naming-clash risk with `platform_proofs/scenarios/`:**
  - Proof scenarios use the same slug space (`scripts/proof/create_scenario_proof.py:18-19`) and the CON-R-06 lifecycle frontmatter (`scripts/proof/scenario_lifecycle.py:24`).
  - Mitigation 1: SFX rejects a slug that already exists under `platform_proofs/scenarios/`.
  - Mitigation 2: SFX never writes under `platform_proofs/` and never names a contract `ScenarioSpec*`.
  - Mitigation 3: generated packages use the `_scenario` infix, so the two trees cannot share a directory name.
- **HO-4 gap:** `scenario_architecture_conformance` resolves packages only under `platform_proofs/scenarios/<slug>` (`scripts/proof/scenario_architecture_conformance.py:157`, `:632`). As it stands, it cannot validate an `applications/` package. HO-4 needs either a conformance entry point for SFX packages (a `scripts/` change in SFX-P9, post-freeze) or an equivalent SFX check. This is **OQ-002-2**.

### D5. Persistence mapping (master plan R1 §7.5, verified at `eadb1c2116fe`)

| Record | Owner (existing) and citation | Versioning | Lifecycle / state | Tenant scope | Retention |
|---|---|---|---|---|---|
| Scenario container | `WorkItem` (`intergrax/contracts/collaborative_work.py:868`; `revision` `:879`) via `WorkItemRepository` (`intergrax/collaborative_work/repository.py:893`) | `revision` | `WorkItemState` OPEN→ACTIVE→COMPLETED/CANCELLED (`collaborative_work.py:759`), coarse | tenant + workspace | business class (ADR-PDC-08 business artifacts → CW, `ADR_PDC_08_ARTIFACT_LIFECYCLE_TAXONOMY.md:19`) |
| Session | `SessionManager` (`intergrax/runtime/nexus/session/session_manager.py:44`) over `SessionStorage` (`intergrax/memory/contracts/session_storage.py:14`); `ChatSession.metadata` (`intergrax/memory/contracts/chat_session.py:29`, field `:48`) = {`sfx_scenario_id`, `work_item_id`} | none (conversation) | `SessionStatus` OPEN/CLOSED (`chat_session.py:16`) | tenant | `MemoryControlPlane` retention sweep (`ADR_PDC_02_CONTEXT_TRIM_FIDELITY_AND_MEMORY_RETENTION.md:23`) |
| Session resume state (SFX-CON-12) | `WorkArtifact` "sfx-session-state" (`collaborative_work.py:1238`) | append-only versions | current pointer (CAS) | tenant + workspace | business |
| Spec, blueprint, test plan, integration plan (SFX-CON-01..07) | `WorkArtifactRepository` (`repository.py:1284`), `WorkArtifactVersionRepository` (`repository.py:1311`), `ArtifactPublicationRepository` (`repository.py:1338`), `ArtifactContentStore` (`intergrax/collaborative_work/content_storage.py:105`) | immutable `WorkArtifactVersion` (`collaborative_work.py:1283`); `publish_version` CAS; content digest = canonical digest (D3) | via the lifecycle record | tenant + workspace | business; never deleted while referenced |
| SFX lifecycle (SFX-CON-14) | `WorkArtifact` "sfx-lifecycle"; each event is a new `WorkArtifactVersion` | append-only | FSM D7; gated transitions require a decision binding | tenant + workspace | audit-grade, never deleted (`ADR_PDC_08_ARTIFACT_LIFECYCLE_TAXONOMY.md:22`) |
| Approval (SFX-CON-13) | `CollaborativeDecisionBindingRepository` (`repository.py:1414`) binding a `WorkArtifactVersionRef` (`collaborative_work.py:1355`) to a `DecisionProposalRef` (`intergrax/contracts/decision_record.py:223`); runtime pauses go through `HumanDecisionPersistence` (`intergrax/runtime/human/persistence_contract.py:16`) | append-only (binding `create` only; decisions insert-only, `persistence_contract.py:25`) | proposed→approved/rejected (in the Decision record) | tenant + workspace | audit evidence, never deleted (`ADR_PDC_08_ARTIFACT_LIFECYCLE_TAXONOMY.md:22`) |
| Engineering job (SFX-CON-15) | `Assignment` via `AssignmentRepository` (`repository.py:925`) + `WorkItemExecutionLink` (`collaborative_work.py:1140`) via `WorkItemExecutionLinkRepository` (`repository.py:1014`); scaffold job execution through `TaskCheckpoint` (`intergrax/runtime/long_running/models.py:19`) in `TaskCheckpointPersistence` (`intergrax/runtime/long_running/persistence_contract.py:45`, CAS `save` `:49`); agentic job state **remains in SEF** CON-N-02 (referenced) | checkpoint `revision` (`models.py:29`) | QUEUED/RUNNING/WAITING/SUCCEEDED/FAILED/DEAD_LETTERED (mapping in D7) | tenant | per long-running policy; scheduler ledger `SchedulerLedger` (`persistence_contract.py:64`) |
| Run facts | Evidence plane `EvidencePersistencePort` (`intergrax/contracts/execution_evidence/persistence_port.py:23`, `append` `:32`) | append-only, idempotent on `event_id` | none | tenant | evidence class |
| Run / ROI report (SFX-CON-09/10) | `WorkArtifactVersion` with `execution` provenance (`collaborative_work.py:1299`; `ExecutionProvenanceRef` `intergrax/contracts/execution_provenance.py:22`) → evidence refs | immutable | none | tenant + workspace | business |
| Catalog sidecar (SFX-CON-11) | `WorkArtifactVersion` keyed by `ApplicationPackage.package_id` (`intergrax/applications/contracts/application_package.py:61`) + `version` (`:63`) + package checksum (`ApplicationDistribution.checksum`, `:52`) | immutable | listed / deprecated / retired (inside the sidecar) | tenant + workspace | business |

**Limits.** No new store and no new state engine. The CW qualification state is REQUALIFICATION_REQUIRED (SFX-P1-03). `ApplicationDistribution.checksum` defaults to an empty string (`application_package.py:52`), so SFX requires a non-empty `sha256:` checksum. Its source is open: **OQ-002-3**.

### D6. Lifecycle-record artifact pattern (append-only versions + CAS pointer)

1. **Identity.** `WorkArtifact` has no kind field (`collaborative_work.py:1238`). SFX artifact kinds are therefore encoded in (a) a deterministic `work_artifact_id` convention `sfx:<scenario_id>:<kind>`, for example `sfx:<id>:lifecycle` and `sfx:<id>:session-state`, and (b) the content media type `application/vnd.intergrax.sfx.<name>.v1+json` in `ArtifactContentRef.media_type` (`collaborative_work.py:1220`). Readers check both and fail closed on a mismatch.
2. **Create.** The first event (`from: null`, `to: DRAFT`) is written with `create_artifact_with_initial_version` (`repository.py:1124`, `:1338`).
3. **Append.** Every later event is a new `WorkArtifactVersion` published with `publish_version`, with `expected_revision` equal to the revision just read (`repository.py:1211`). A CAS conflict is never retried blindly. The writer re-reads the current version, re-validates the transition against D7 and either republishes or returns `sfx.lifecycle.conflict`. Nothing is overwritten.
4. **Current state.** The current state is the record at `WorkArtifact.current_version_id` (`collaborative_work.py:1249`). History is `list_for_artifact` (`repository.py:1311`), ordered by `(published_at, work_artifact_version_id)`.
5. **Chain.** Each record carries `sequence` (= `version`) and `previous_record_digest`, so a missing or reordered version is detectable without trusting store order.
6. **Gated transitions.** `BLUEPRINTED→APPROVED` and `SANDBOX_TESTED→SANDBOX_CERTIFIED` require `decision_binding_ref` and a HUMAN actor. The writer resolves the binding through `CollaborativeDecisionBindingRepository.get` (`repository.py:1414`). It checks that the bound `work_artifact_version` (`intergrax/contracts/collaborative_decision_binding.py:75`) is the exact approved version, whose digest equals the approved digest, and that the referenced decision is approved. The binding is an association that "never becomes" an authority (`collaborative_decision_binding.py:66`).
7. **Artifact revisions.** An edit to another artifact is logged as an `ARTIFACT_REVISION` event with `from == to`, a `subject_artifact_ref` and an `intervention_class`. These events feed G-2 and G-4 (`SFX_METRICS.md`). This is an SFX extension of plan §7.5, which lists only transitions: **OQ-002-4**.

### D7. FSM and mapping to `scripts/proof/scenario_lifecycle.py`

The SFX states are those of master plan R1 §7.9. The transitions are encoded in `sfx.scenario_lifecycle_record.v1`. `ScenarioLifecycle` values verified at `scripts/proof/scenario_lifecycle.py:24-29`:

| SFX state | `ScenarioLifecycle` projection |
|-----------|--------------------------------|
| DRAFT, DISCOVERED, SPECIFIED, BLUEPRINTED | `DESIGN` |
| APPROVED, BUILDING | `ACCEPTED_FOR_IMPLEMENTATION` (BUILDING keeps the APPROVED projection) |
| BUILT | `IMPLEMENTATION_INITIALIZED` |
| SANDBOX_TESTED | `EXECUTABLE` |
| SANDBOX_CERTIFIED, ADMISSION_DOSSIER_READY | `VERIFIED` (dossier export adds no lifecycle claim) |

- The projection is **informational** (`legacy_lifecycle_projection`). The authoritative state is the CW lifecycle record, because the markdown frontmatter is hand-editable (master plan R1 §14.2).
- SFX never writes lifecycle frontmatter into `platform_proofs/`.
- `VERIFIED` in this projection is not a production claim (SFX-INV-01).

**Engineering job states** (SFX-CON-15) over `TaskState` (`intergrax/runtime/task/task_state.py:11`), for scaffold jobs:

| SFX job state | `TaskState` values |
|---------------|--------------------|
| QUEUED | CREATED, CLASSIFIED, PLANNED |
| RUNNING | RUNNING, VALIDATING |
| WAITING | WAITING_FOR_RESOURCES, WAITING_FOR_HUMAN, NEEDS_MORE_INFORMATION |
| SUCCEEDED | COMPLETED |
| FAILED | FAILED, CANCELLED, EXPIRED, PARTIALLY_COMPLETED (a partial build fails closed) |
| DEAD_LETTERED | none at the reference SHA (see below) |

- `DEAD_LETTERED` has no `TaskState` at the reference SHA. It maps to the ADR-PDC-07 terminal queue state `DEAD_LETTERED` (`ADR_PDC_07_BACKGROUND_RECURRENCE_AND_DLQ.md:21`) once PDC-X delivers it. Until then SFX cannot report it (**OQ-002-5**).
- Agentic jobs follow HO-5: SEF `failed` maps to `FAILED` and SEF `paused` maps to `WAITING`.
- A FAILED job moves the scenario from `BUILDING` back to `APPROVED`, with no automatic retry beyond `max_attempts`.

### D8. Why `ApprovalStorePort` / `InMemoryApprovalStore` is not used

- **Wrong semantics.** `ApprovalStorePort` (`intergrax/contracts/agent_runtime_governance.py:366`) persists `ApprovalRequest` (`:253`), the HITL contract for one governed **tool execution**. It wraps a `ToolAuthorizationRequest` (`:260`) and has an `expires_at` (`:264`). An SFX approval binds an exact artifact version and its digest. It is not a tool call.
- **Mutable.** The port exposes `update` (`:373`). `InMemoryApprovalStore` (`intergrax/runtime/agent_governance/approval_boundary.py:18`) overwrites the stored request in place (`approval_boundary.py:34`). SFX approvals are audit evidence that must be append-only and never deleted (`ADR_PDC_08_ARTIFACT_LIFECYCLE_TAXONOMY.md:22`).
- **Not durable or scoped.** The only implementation is an in-process dict (`approval_boundary.py:22`), and `get(approval_id)` has no tenant or workspace scope (`agent_runtime_governance.py:371`).
- **No second authority.** Repurposing it would give SFX a second approval authority next to the decision record. SFX instead uses `CollaborativeDecisionBindingRepository` (append-only `create`, `repository.py:1414`) for artifact approvals. During L2 runs it uses the existing `HumanDecisionPersistence` path (`intergrax/runtime/human/persistence_contract.py:16`) for runtime pauses.

## Consequences

### Positive

- No new store, state engine or approval authority. Every SFX record has an existing owner, tenant scope and retention class.
- One canonical byte form shared by the CW content digest, the approval binding, the handoff (`D_bp`) and door parity.
- The lifecycle history is tamper-evident (CAS plus a digest chain) and audit-grade.

### Negative

- RFC 8785 needs an implementation or a dependency, which the platform does not have today (OQ-002-6).
- Workspace naming differs from the plan text unless the scaffold changes (OQ-002-1).
- HO-4 conformance needs an SFX entry point (OQ-002-2).
- `DEAD_LETTERED` depends on PDC-X (OQ-002-5).
- SFX artifact kind is a naming convention over CW, not a typed field. Readers must enforce it.

## Compliance

- Tier boundaries: all contracts live in Tier-2 `agents/scenario_factory/`, and nothing in `intergrax/` imports them.
- No product code at this ADR. Schemas are design drafts until SFX-P5, which needs ARCH-FREEZE CLOSED and GATE-SFX-ENTRY.
- The status stays **Proposed** until OD-SFX-06.
- Secrets appear only as references (env/vendor qualification rule).

## Open questions (operator or architect decision)

| ID | Question | Proposed answer |
|----|----------|-----------------|
| OQ-002-1 | Workspace path: `applications/<slug>_scenario/` (needs a scaffold change) or the scaffold output `applications/<slug>_scenario_application/` | Scaffold output; no Tier-0 change |
| OQ-002-2 | HO-4: extend `scenario_architecture_conformance` to `applications/` packages, or an SFX conformance check | Extend through a `scripts/` entry point in SFX-P9 |
| OQ-002-3 | Source of `package_checksum`: `ApplicationDistribution.checksum` (may be empty) or an SFX-computed tree digest | SFX-computed `generated_tree_digest`, required to equal a non-empty distribution checksum |
| OQ-002-4 | Log artifact revisions in the lifecycle stream (`ARTIFACT_REVISION`) for G-2/G-4 | Yes |
| OQ-002-5 | `DEAD_LETTERED` before PDC-X D7 lands | Report FAILED with `failure.code = sfx.job.retry_budget_exhausted` until then |
| OQ-002-6 | RFC 8785 implementation: dependency (needs the dependency allowlist, OD-SFX-14 scope) or an in-repo Tier-2 implementation | In-repo implementation, cross-checked against an independent implementation in tests |

## Implementation notes

- **Not in scope at Proposed:** pydantic models, generated schemas, repositories or adapters (SFX-P5+).
- **Verification when implemented:** PC-1..PC-6 tests (`SFX_CANONICALIZATION.md`), lifecycle CAS-conflict tests, gated-transition negative tests, and `check_harness_adr.py` after the index update.
- **Revision triggers:** a change to the CW artifact contracts, acceptance of SEF CON-N/CON-E contracts with different fields, or a PDC-X change to queue states.
