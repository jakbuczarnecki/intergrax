# PLAN T-IGX-093-A — AHE W1-B01 canonical ADAS contracts and candidate archive

Roadmap package **AHE-W1-01**. Canonical obligations **AHI-ADAS-10.1** through **AHI-ADAS-10.14**.

Deliver the ADAS typed contract and append-only candidate archive foundation. Implement exactly the configuration-only LAB behavioral profile. No execution engine, strategy implementation, static gate implementation, evaluation runner, production activation or promotion behavior belongs to this batch. All existing ADAS owners remain authoritative.

## Purpose and scope

This plan defines the executable implementation contract for W1-B01. It is an approval artifact and must remain stable after operator approval. Neither OD-AHE-01 nor OD-AHE-02 constitutes approval of this plan.

## Future authorized write set (implementation candidate only)

### Runtime files

- `intergrax/runtime/adaptive/agent_design_search/__init__.py`
- `intergrax/runtime/adaptive/agent_design_search/contracts.py`
- `intergrax/runtime/adaptive/agent_design_search/archive.py`
- `intergrax/runtime/adaptive/agent_design_search/sqlite_archive.py`

### Focused test files

- `tests/unit/runtime/adaptive/agent_design_search/test_contracts.py`
- `tests/unit/runtime/adaptive/agent_design_search/test_archive.py`
- `tests/unit/runtime/adaptive/agent_design_search/test_sqlite_archive.py`

### Canonical plan register updates (evidence and obligation tracking only)

- `docs/project/maintainers/plans/ADAPTIVE_HARNESS_INTELLIGENCE.md`
- `docs/project/maintainers/plans/ADAPTIVE_HARNESS_INTELLIGENCE_agent_design_search.md`
- `docs/project/maintainers/plans/satellites/ADAPTIVE_HARNESS_INTELLIGENCE_agent_design_search.md`
- `docs/project/maintainers/plans/satellites/ADAPTIVE_HARNESS_EVOLUTION.md`

The future candidate may update these plan registers only to record implementation evidence, exact touched obligations and residuals. No independent acceptance may be declared by the implementer.

Do not modify the parent adaptive runtime package exports. Do not modify existing profile stores.

Governance ticket lifecycle and publication authorization records are external authority artifacts. They must not be changed inside the product candidate.

## Contract implementation requirements

Use Pydantic v2 and Python 3.12. Preserve canonical ADAS §13.0–§13.7 contract identities and serialized field meanings.

Implement:

- `AdasOperationalEnvelope`
- `AgentDesignObjective`
- `AgentDesignSearchRun`
- `AgentCandidateDraft`
- `AgentCandidateStatus`
- `AgentCandidateRecord`
- `AgentCandidateEvaluationResult`
- `AgentCandidateEvidenceBundle`

Additional small typed value objects may be defined inside `contracts.py` for tenant/workspace scope binding, immutable baseline configuration references, single configuration delta, aggregate search limits, resource accounting completeness, and selection outcome. They do not constitute separate platform authorities.

Reuse existing `ProfileArtifactType` and `MergeStrategy` rather than defining competing enums.

### General invariants

- Reject unknown fields.
- Reject blank required identities.
- Require explicit tenant and workspace scope.
- Reject inconsistent duplicated scope identities.
- Require timezone-aware timestamps.
- Reject nonfinite numeric values.
- Reject negative counts, durations and resource consumption.
- Validate digest format as lowercase 64-character hexadecimal SHA-256.
- Preserve canonical field names.
- Prevent mutation of nested collections and nested JSON payloads.
- Preserve canonical JSON object and array representation during serialization.
- Do not use unrestricted executable objects, arbitrary callables or dynamic imports as contract payloads.
- Do not treat successful contract validation as runtime admission or approval.
- Frozen model configuration alone is insufficient proof of deep immutability.

### Operational envelope

Retain the fields defined in canonical ADAS §13.0. Add explicit tenant and workspace identity, accepted scope reference and LAB-only posture.

For W1, require: `schema_version`, `created_at`, `created_by`, `tenant_id`, `workspace_id`, `correlation_id`, `trace_id`, `policy_context_id`, `accepted_scope_ref`, `retention_policy`.

Retain optional approval, audit, idempotency, artifact, source and evaluation references. Approval-related fields are provenance only. A caller-supplied approval reference never grants authority.

### AgentDesignObjective

Retain canonical objective fields and apply W1 requirements: explicit operational envelope; scope consistency; task class and target capability; immutable baseline version reference; expected artifact type; baseline configuration digest; dataset identifier and content digest; acceptance criteria reference; explicit admitted configuration surface; allowed configuration alternatives; aggregate resource limits; external effects prohibited; human approval requirement preserved.

For W1, constrain: maximum candidates 1; maximum search iterations 1; maximum root executions 2; finite aggregate token allowance; finite aggregate monetary allowance; finite deadline.

The W1 configuration surface is merge strategy. Use the existing serialized `MergeStrategy` values. The planned behavioral witness uses `last_wins` and `concat`. No model, tool, permission, capability or policy mutation is authorized.

### AgentDesignSearchRun

Preserve canonical fields and add operational envelope and scope binding; immutable baseline reference; canonical execution references; explicit aggregate accounting evidence references; selection outcome when determined; structured stop reason.

Do not use default zero to represent consumption that has not been measured. A terminal search must preserve its completion timestamp and stop reason. A failed or cancelled run must not claim successful selection.

### AgentCandidateDraft

Preserve canonical field names and lineage semantics. Add search-run identity; explicit scope; baseline reference and digest; candidate configuration digest; exactly one typed configuration delta.

For W1, the delta contains the existing admitted merge-strategy value and its replacement. Reject no-op deltas. Reject executable source payloads. Reject permission expansion.

Scaffold-specific fields that cannot exist in a configuration-only experiment shall be explicitly absent. Preserve their canonical names. Do not fabricate scaffold patterns, scaffold results or source artifact references. Existing capability and agent-contract references may be preserved as immutable provenance, not as executable drafts.

### AgentCandidateStatus

Preserve all canonical members and serialized values exactly:

- DRAFT = draft
- SCAFFOLDED = scaffolded
- STATIC_GATE_FAILED = static_gate_failed
- STATIC_GATE_PASSED = static_gate_passed
- EVALUATED = evaluated
- REJECTED = rejected
- SHADOW = shadow
- CANARY = canary
- PROMOTED = promoted
- RETIRED = retired
- ARCHIVED = archived

Do not replace them with a W1-specific enum.

W1 archive transitions: DRAFT to STATIC_GATE_PASSED; DRAFT to STATIC_GATE_FAILED; DRAFT to REJECTED; STATIC_GATE_PASSED to EVALUATED; STATIC_GATE_PASSED to REJECTED; EVALUATED to REJECTED when explicitly supported by rejection evidence; applicable terminal historical states to ARCHIVED only through an explicit lifecycle operation. Archiving a snapshot does not automatically change its status.

Do not permit transitions to scaffolded, shadow, canary, promoted or retired states in the W1 configuration-only archive profile.

Selection outcomes are separate: SELECT_CANDIDATE; RETAIN_BASELINE; ABSTAIN.

### AgentCandidateRecord

Retain all canonical fields. Add explicit envelope, scope, search-run identity, configuration references and immutable lineage binding. Use immutable historical snapshots.

Identity invariants across snapshots: candidate ID; objective ID; search-run ID; tenant; workspace; parent candidate ID; lineage ID; baseline identity and digest; candidate configuration digest. Preserve creation provenance. Allow new evidence and lifecycle status only through validated append operations. Scaffold and promotion references not applicable to W1 must remain explicitly absent.

### AgentCandidateEvaluationResult

Preserve canonical metric names including quality; normalized cost; latency; tokens; steps; tool calls; LLM calls; HITL interventions; regression flags; utility; baseline utility; utility delta; golden pass rate; negative pass rate; adversarial pass rate; failure reasons.

Add canonical execution and evaluation references and evidence-completeness information. Metrics not measured or not applicable must be explicitly absent. No synthetic zero substitution. A value of zero is a measured value only when supported by the relevant evidence. A successful evaluation verdict requires all mandatory W1 evidence.

This batch implements the data contract only. Actual evaluation and comparison behavior remain deferred to W1-B02.

### AgentCandidateEvidenceBundle

Preserve canonical bundle fields and nested record identities. Add explicit workspace, configuration digests, execution references, accounting references and selection outcome where applicable.

Support sealed terminal evidence for static rejection; failed execution; successful evaluation; selection; baseline retention; abstention.

A rejected candidate need not have evaluation results that could not have been produced. Represent absence explicitly; never fabricate evaluation data. All supplied evidence references must agree with the bundle scope and candidate lineage. A sealed bundle must be deeply immutable.

Compute SHA-256 over deterministic canonical JSON, excluding only the self-referential digest field. Define and test canonical handling of key order; enum values; UTC timestamps; optional values; nested objects and arrays; nonfinite numeric rejection. Digest verification proves content integrity, not authority or correctness.

## Archive public contract

Define `AgentDesignArchive` as a typed protocol. Provide equivalent implementations: `InMemoryAgentDesignArchive` and `SQLiteAgentDesignArchive`.

Expose operations for: appending a candidate snapshot; retrieving the latest candidate snapshot; retrieving candidate history; listing candidates for a scoped objective; filtering by candidate status; retrieving parent and child lineage; appending a sealed evidence bundle; retrieving a sealed evidence bundle.

All read operations require explicit tenant and workspace scope. All append operations must bind the supplied scope to the record or bundle scope. Unknown or foreign identity lookups must not expose foreign record contents or distinguish foreign records through detailed error payloads. Do not provide an unscoped list operation. Archive operations must not dispatch tasks or invoke runtime execution.

## Append-only and lineage invariants

The archive must: preserve immutable historical snapshots; preserve deterministic history ordering; reject invalid lifecycle transitions; reject changes to immutable identity or lineage fields; reject parent references crossing tenant, workspace or objective boundaries; reject self-parenting and cyclic lineage; reject conflicting reuse of the same snapshot identity; support idempotent exact duplicate append; preserve all historical evidence; reject sealed evidence bundle replacement; reject inconsistent evidence digests; prevent mutation through caller-held or returned data references.

Candidate identity is stable across snapshots. Snapshot identity and history ordering must be explicit. A candidate ID must not be rebound to a different tenant, workspace, objective or configuration lineage. No archive operation may silently overwrite historical data.

## SQLite implementation

Use Python standard-library `sqlite3`. Require an explicit database path. Do not create the database on module import. Do not initialize a default production database. Use parameterized SQL. Use transactions for identity validation, transition validation and append operations. Enforce database uniqueness constraints for snapshot and bundle identities. Provide deterministic ordering independent of timestamp collisions. Make append validation and write atomic under concurrent access. Use the same canonical serialization as the in-memory implementation. Provide explicit resource cleanup.

Reopening the database must preserve candidate identities; latest snapshots; historical snapshots; lineage; evidence bundles; digests; idempotency semantics.

Schema creation is scoped to the explicit ADAS LAB database. No production migration, destructive schema reset, purge or retention worker belongs to W1-B01.

## Mandatory tests

`test_contracts.py` must verify: canonical contract names and serialized fields; canonical candidate status values; selection/lifecycle separation; valid W1 objective construction; invalid scope rejection; missing mandatory provenance rejection; invalid or missing baseline digests; invalid budget rejection; nonfinite number rejection; invalid and no-op configuration deltas; deep immutability; deterministic canonical serialization; evidence digest stability and mismatch detection; explicit unknown-metric behavior; explicit absent scaffold and promotion fields; evidence for rejected or failed candidates without fabricated evaluation results.

`test_archive.py` and `test_sqlite_archive.py` must execute equivalent archive contract cases: valid snapshot append; ordered history; latest record retrieval; exact duplicate idempotency; conflicting duplicate rejection; invalid transition rejection; immutable lineage rejection; cross-tenant read and write rejection; cross-workspace read and write rejection; cross-objective lineage rejection; self-parent and cyclic lineage rejection; scoped objective and status queries; parent/child retrieval; sealed bundle append and retrieval; conflicting bundle rejection; digest mismatch rejection; mutation attempts through input and returned objects.

SQLite-specific tests: close and reopen durability; persistence of history and lineage; persistence of sealed bundle integrity; atomic append failure behavior; concurrent conflicting append behavior; explicit temporary database path; no default database creation on import.

Package import tests must demonstrate no provider calls, execution dispatch, promotion or database initialization. Do not weaken existing tests.

## Test and qualification contract

Mandatory focused regression:

`uv run --no-sync pytest -q tests/unit/runtime/adaptive/agent_design_search/`

Mandatory adjacent regression:

`uv run --no-sync pytest -q tests/unit/runtime/adaptive/ --ignore=tests/unit/runtime/adaptive/agent_design_search/`

Mandatory scoped lint:

`uv run --no-sync ruff check intergrax/runtime/adaptive/agent_design_search/ tests/unit/runtime/adaptive/agent_design_search/`

Mandatory scoped typing:

`uv run --no-sync pyright intergrax/runtime/adaptive/agent_design_search/`

Mandatory compilation:

`uv run --no-sync python -m compileall -q intergrax/runtime/adaptive/agent_design_search/`

Mandatory scoped diff validation and applicable documentation checks.

Mandatory governance quality:

`uv run --no-sync python -m scripts.governance.igx_gov ticket`

Execute commands sequentially. Preserve command, exit-code and test-count evidence. Use repository-sanctioned Python 3.12 and existing dependencies. Do not add dependencies or modify environment policy. Temporary evidence belongs under the existing `.tmp/` convention.

## Canonical obligations and publication

Map each implementation result to AHI-ADAS-10.1–10.14. For every obligation, record implementation status; evidence references; actual qualification; canonical residuals; independent acceptance pending.

Preserve canonical residuals associated with executable agent scaffolding; full canonical promotion evidence; non-W1 lifecycle behavior; enterprise retention; production persistence integration.

Do not automatically declare AHI-ADAS-10 fully accepted.

The implementation candidate is a single scoped AHI-ADAS-10 PR targeting development. Use the existing `CHATGPT_DIRECT_IMPLEMENTATION` candidate publication route only after its exact external authorization exists. The future implementation ticket must be approved and present on the authorized base. The future exact write set must be registered by the external authority. The implementer must not create or alter its own publication authorization.

The implementer commits the candidate with exactly one canonical ticket Ref trailer. The implementer publishes the candidate through the approved publisher. Independent review and applicable integration requirements remain mandatory. Do not bypass the canonical ADAS PR delivery rule.