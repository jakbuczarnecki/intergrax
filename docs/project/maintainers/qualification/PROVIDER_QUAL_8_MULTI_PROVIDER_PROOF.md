# PROVIDER-QUAL-8 — Multi-Provider Real Proof

**Status:** `READY_FOR_REVIEW`

**Baseline ancestor:** `dd5632e8c97c4dd4180c4f073ec356321d5d9059`

## Goal

Prove that the provider qualification architecture closed in PROVIDER-QUAL-7 is genuinely multi-provider and capability-oriented — the same qualification-core runner executes against multiple real provider implementations without vendor dispatch in `intergrax/core/qualification/`.

## Provider inventory

| provider_id | capability/domain | real adapter | Integrations catalog | CW typed materialization | real backend proof env | qualification suite | production qualification feasible |
|-------------|-------------------|--------------|----------------------|--------------------------|------------------------|----------------------|-----------------------------------|
| `postgresql` | `collaborative_work.persistence.v1` | YES | YES (`register_postgresql_integration`) | YES (`bind_collaborative_work_materialization`) | YES (`infra/docker/postgresql`, DSN env) | `cw.postgresql.repository.v1` | YES (`PRODUCTION_QUALIFIED`) |
| `sqlite` | `collaborative_work.persistence.v1` | YES | YES (`register_sqlite_integration`) | YES (`bind_collaborative_work_materialization`) | YES (local lab `data_dir`) | `cw.sqlite.repository.v1` | YES (`QUALIFIED` — local lab) |
| `mysql` | relational store only | YES | YES (`register_mysql_integration`) | **NO** | YES (`infra/docker/mysql`) | — | **NO** — missing CW binder |
| `oracle` | relational store only | YES | YES (`register_oracle_integration`) | **NO** | partial (no CW harness) | — | **NO** — missing CW binder |
| `mongodb` | document store | YES | YES | N/A (different category) | YES (mongo integration tests) | — | **NO** — no CW qualification contract |

### Platform reuse classification

| Provider | Classification | Notes |
|----------|----------------|-------|
| PostgreSQL | A. READY_USING_EXISTING_PLATFORM | Full PROVIDER-QUAL-3B materialization + PROVIDER-QUAL-7 runner |
| SQLite | A. READY_USING_EXISTING_PLATFORM | Same shared semantic suite, distinct suite ID + status |
| MySQL | B. NEEDS_NARROW_EXISTING_PROVIDER_EXTENSION | Requires `bind_collaborative_work_materialization` adapter only |
| Oracle | B. NEEDS_NARROW_EXISTING_PROVIDER_EXTENSION | Same as MySQL |
| Others (vector/document) | C. REQUIRES_NEW_PLATFORM_MECHANISM | Different capability domain; out of scope |

**Third real provider:** not included in this proof. MySQL and Oracle have relational adapters and Docker infra but lack Collaborative Work typed materialization (`bind_collaborative_work_materialization`). Adding them is a narrow provider extension (B), not a qualification-core change — deferred to a follow-on provider implementation task.

## Abstraction flow (unchanged from PROVIDER-QUAL-7)

```text
IntegrationProfile
  → resolve_integration_provider_id (generic, no vendor branch)
  → ProviderQualificationDomainBinding.validate_resolved_provider
  → ProviderQualificationDomainBinding.materialize
  → resolve_collaborative_work_repositories (PROVIDER-QUAL-3B)
  → CollaborativeWorkRepositoryQualificationSuite.execute (shared semantic checks)
  → ProviderQualificationRun
  → DocumentStoreProviderQualificationPersistence
  → discovery / reload
```

Qualification core (`execution.py`, `suite.py`, `observability.py`) contains **zero** vendor-specific execution logic.

## Real provider matrix

| Provider | Capability | Backend | Materialization | Suite | Status | Evidence |
|----------|------------|---------|-----------------|-------|--------|----------|
| PostgreSQL | CW persistence | real Docker PG | `resolve_collaborative_work_repositories` → PG binder | `cw.postgresql.repository.v1` | `PRODUCTION_QUALIFIED` | `test_provider_qualification_execution_postgresql.py` |
| SQLite | CW persistence | real local SQLite | `resolve_collaborative_work_repositories` → SQLite binder | `cw.sqlite.repository.v1` | `QUALIFIED` | `test_provider_qualification_multi_provider_proof.py` |

**Architecture-only extension proof:** `test_synthetic_third_provider_requires_no_core_vendor_dispatch_changes` (unit) — synthetic binding, not a production claim.

## Semantic suite sharing

PostgreSQL and SQLite use the same `CollaborativeWorkRepositoryQualificationSuite` class and `_run_repository_contract_checks` implementation. Suite IDs differ (`cw.postgresql.repository.v1` vs `cw.sqlite.repository.v1`) for provenance; provider identity lives in `ProviderQualificationSubject`, not duplicated semantic logic.

**Architecture finding:** future migration to `cw.repository.contract.v1` with provider identity only in subject is feasible but not required for this proof.

## Provider version provenance

`ProviderQualificationSubject.provider_version` is supplied explicitly by the execution harness (test/integration config). Platform does not introspect backend version at qualification-core layer. PostgreSQL integration proof uses execution-config version string; SQLite uses `"execution-config-lab"`. **Gap:** no canonical Integrations metadata for live backend version introspection.

## Observability / diagnostics

Classification unchanged from PROVIDER-QUAL-7-R2:

- Observability: `REUSE_EXISTING_PLATFORM_SIGNAL_OBSERVABILITY`
- Diagnostics: `PLATFORM_PROBLEM_SIGNAL_ONLY_DIAGNOSTICS_DOWNSTREAM_BY_HOST`

No vendor-specific telemetry. Infrastructure failures emit canonical `PlatformProblemSignal` family.

## Failure matrix (proved)

| Scenario | PostgreSQL | SQLite |
|----------|------------|--------|
| A. provider available → executes | integration test | multi-provider proof |
| B. bad/missing config → explicit failure | skip when DSN missing | `test_bad_config_materialization_failure_explicit` |
| C. no silent fallback | vendor abstraction gate + resolution tests | `test_wrong_provider_id_fails_closed` |
| D. wrong provider_id vs subject → fail closed | subject mismatch tests (unit) | multi-provider proof |
| E. semantic failure → REJECTED | suite semantics tests (unit) | shared suite |
| F. persistence failure separate | persistence unit tests | idempotent recovery proof |

## Requalification composition

```text
prior ProviderQualificationRun (sqlite)
  → validity STALE or REVOKED
  → establish_provider_requalification_requirement
  → prepare_provider_requalification_run_identity
  → execute_provider_qualification (shared runner, new run ID)
  → second immutable ProviderQualificationRun
```

Proved in `test_sqlite_requalification_composes_with_shared_runner` and `test_revoked_prior_run_remains_terminal_new_qualification_succeeds`.

REVOKED prior run remains terminal forever; new run succeeds independently.

## Commands

```bash
uv run pytest tests/unit/core/qualification/ -q
uv run pytest tests/integration/core/qualification/test_provider_qualification_multi_provider_proof.py -q
uv run pytest tests/integration/core/qualification/test_provider_qualification_execution_postgresql.py -m "integration and network" -q
```

PostgreSQL requires Docker PostgreSQL and `INTERGRAX_COLLABORATIVE_WORK_POSTGRESQL_DSN` or `INTERGRAX_POSTGRESQL_*` settings.

## Core changes

Qualification core vendor dispatch changes: **0**.

## Security

DSNs/passwords not persisted in qualification evidence. Provider config stays provider-owned via `IntegrationProfile.options`.

## Limitations

- Only PostgreSQL + SQLite proven for Collaborative Work persistence qualification.
- MySQL/Oracle require narrow provider binder extension before qualification proof.
- SQLite status remains `QUALIFIED` (local lab), not `PRODUCTION_QUALIFIED`.

## Future provider onboarding

1. Implement `bind_collaborative_work_materialization` on the provider factory (PROVIDER-QUAL-3B contract).
2. Add domain binding factory with provider-specific suite ID + environment metadata (reuse `CollaborativeWorkRepositoryQualificationSuite`).
3. Add integration proof test using shared `execute_provider_qualification` — no qualification-core changes.
