# PROVIDER-QUAL-7 — Shared Provider Qualification Execution Runner

**Status:** `READY_FOR_REVIEW` (PROVIDER-QUAL-7-R2)

## Scope

Synchronous, provider-neutral qualification execution coordinator:

- `execute_provider_qualification(request, dependencies)` → `ProviderQualificationRun`
- No scheduler, worker queue, or CI framework
- No vendor-specific central dispatch in qualification core
- Reuses Integrations provider resolution and PROVIDER-QUAL-3B typed domain materialization
- Reuses PROVIDER-QUAL-3C persistence (`DocumentStore` / `ProofReceipt`)
- Reuses platform non-execution observability (`PlatformObservabilityExportSource`) and problem plane (`PlatformProblemSignal`)

## Execution truth vs observability vs diagnostics

```text
Qualification Runner
        ↓
ProviderQualificationRun construction
        ↓
Qualification Persistence          ← authoritative qualification truth

        ↘ (best-effort)
         Platform Signal / PlatformProblemSignal
                    ↓
             HOS / export policy (host composition)
                    ↓
       central diagnostics when host wires ProblemPersistence
```

**Frozen principle:** execution creates truth; observability records/projects truth; diagnostics interprets truth. Observability failure must never alter qualification execution outcome or persisted `ProviderQualificationRun` facts.

## Lifecycle

```mermaid
flowchart TD
    A[Evidence becomes STALE or REVOKED] --> B[ProviderRequalificationDecision]
    B --> C[prepare_provider_requalification_run_identity]
    C --> D[ProviderQualificationExecutionRequest]
    D --> E[execute_provider_qualification]
    E --> F[IntegrationProfile provider resolution]
    F --> G[Typed domain materialization]
    G --> H[Domain qualification suite]
    H --> I[ProviderQualificationRun]
    I --> J[Persistence / discovery]
    J --> K[Separate validity evaluation PROVIDER-QUAL-5]
```

## PLATFORM REUSE

| Mechanism | Reused |
|-----------|--------|
| Integrations provider resolution | YES |
| PROVIDER-QUAL-3B materialization (`resolve_collaborative_work_repositories`) | YES |
| PROVIDER-QUAL-3C persistence (`DocumentStoreProviderQualificationPersistence`) | YES |
| HOS non-execution platform observability (`PlatformObservabilityExportSource`) | YES |
| Platform problem plane (`PlatformProblemSignal` via export envelope) | YES |
| Second registry / telemetry pipeline | NO |
| Second diagnostics store | NO |

## OBSERVABILITY / DIAGNOSTICS

**Observability classification:** `REUSE_EXISTING_PLATFORM_SIGNAL_OBSERVABILITY`

**Diagnostics classification:** `PLATFORM_PROBLEM_SIGNAL_ONLY_DIAGNOSTICS_DOWNSTREAM_BY_HOST`

- canonical `PlatformProblemSignal` reused = **YES**
- central diagnostics ingestion (`ProblemPersistence` → `DiagnosticReadService`) = **host/composition responsibility**
- qualification core does not import `ProblemPersistence` or application composition roots

Qualification execution is not Task/Run/Attempt lifecycle. Lifecycle facts emit through:

```text
ProviderQualificationExecutionEvent
  → PlatformObservabilityExportSource
  → ObservabilityExportEnvelope (PLATFORM_SIGNAL)
```

Infrastructure failures emit through the existing problem export path:

```text
build_qualification_infrastructure_problem_envelope
  → PlatformProblemSignal
  → ObservabilityExportEnvelope (PROBLEM_SIGNAL)
  → host HOS/export wiring (when provided)
  → central diagnostics (when host wires canonical ingestion)
```

Observability is best-effort via `safe_record_qualification_observability` and optional `ProviderQualificationExecutionObservabilityPort`. Export/telemetry failures are logged and degraded per platform export health conventions; they do not alter qualification truth.

`RecordingProviderQualificationExecutionObservability` is a **TEST / LOCAL PROOF** in-memory collector only — not enterprise production diagnostics. `NoOpProviderQualificationExecutionObservability` is acceptable for isolated unit tests and explicitly non-observed lab usage only.

Production qualification execution hosts **must** inject the canonical platform observability adapter (HOS/export/diagnostics composition). No dedicated qualification host exists in core yet; core documents this requirement.

## VENDOR ABSTRACTION

- Qualification runner has no vendor-specific materialization code.
- Collaborative Work qualification binding delegates materialization to `resolve_collaborative_work_repositories(profile)`.
- Provider mechanics (including isolated PostgreSQL `schema_name`) live in Integrations/provider binders (`PostgreSQLRelationalStoreFactory.bind_collaborative_work_materialization`).
- Domain suite uses explicit semantic checks — no authoritative `assert` usage.

## IDEMPOTENCY / CONCURRENCY

- Compatible persisted run (`subject`, `source_revision`, `executor`) → safe idempotent return + best-effort `execution.recovered` observability.
- Incompatible request for same `qualification_run_id` → `ProviderQualificationRequestIncompatibleError`.
- Persistence conflict with different stored fact → `ProviderQualificationExecutionConflictError` (fail closed).
- Identical duplicate after conditional persist conflict → safe recovery when `stored == run`.

## REAL PROVIDER PROOF

Canonical path:

```text
IntegrationProfile (+ provider options such as schema_name)
  → resolve_integration_provider_id
  → CollaborativeWorkRepositoryQualificationBinding.materialize
  → resolve_collaborative_work_repositories
  → PROVIDER-QUAL-3B binder/factory
  → Collaborative Work suite
  → ProviderQualificationRun
  → DocumentStore persistence
```

```bash
uv run pytest tests/unit/core/qualification/test_provider_qualification_execution_runner.py -q
uv run pytest tests/unit/core/qualification/test_provider_qualification_observability_isolation.py -q
uv run pytest tests/integration/core/qualification/test_provider_qualification_execution_postgresql.py -m "integration and network" -q
```

PostgreSQL integration proof requires Docker PostgreSQL and `INTERGRAX_COLLABORATIVE_WORK_POSTGRESQL_DSN` or equivalent `INTERGRAX_POSTGRESQL_*` settings. Isolated qualification schema is supplied through provider profile options (`schema_name`), not domain qualification code.

## Ownership

| Layer | Owns |
|-------|------|
| Platform qualification core | execution coordination, run identity, executor metadata, canonical run construction, persistence coordination, infrastructure failure semantics, optional platform observability port |
| Domain suite | semantic checks, suite identity/version, outcome mapping |
| Provider / Integrations | config, credentials, materialization, backend mechanics |
| Product host / composition | canonical observability export, central diagnostics ingestion |

## First concrete domain

Collaborative Work repository qualification (`intergrax/collaborative_work/repository_qualification_suite.py`) proves PostgreSQL and SQLite execution through injected `ProviderQualificationDomainBinding` — qualification core contains no vendor branches.
