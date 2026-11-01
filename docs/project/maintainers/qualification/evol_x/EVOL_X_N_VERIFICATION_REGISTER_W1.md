# EVOL-X W1 — N-01..N-18 verification register

**Baseline SHA:** `5a4e922ffacabef0ee24eb676e5c8705c20ef4a1`

| ID | Question | Result | Evidence |
|----|----------|--------|----------|
| **N-01** | Fingerprint inputs / semantic scope | **VERIFIED** | `compute_effective_profile_fingerprint` uses `bundle_normalized_payload(profile.bundle_dump(mode="json"))` → `stable_digest_hex` — `intergrax/applications/_shared/profile_resolution/fingerprint.py` |
| **N-02** | EffectiveProfileRevision retention / pin lifecycle | **PARTIAL — decision required** | Pinning/fail-closed verified in `execution_pinning.py`; no dedicated purge/TTL for revision store in profile_resolution grep — retention is store-policy / composition concern → **OD-EVOL-02** documents WM-03 assumption: superseded revisions must remain readable for in-flight |
| **N-03** | `ActivationService` vs durable schema mismatch | **VERIFIED (gap)** | `rollback()` checks serving pointer, SUPERSEDED prior, artifact revalidation, identity, instance readiness — **no** persisted-state/schema compatibility — `activation.py:423–530` |
| **N-04** | `ApplicationGraphSpec` version vs `GraphSpecMigration` | **VERIFIED (contract gap)** | `ApplicationGraphSpec` has **no** `graph_version`; `GraphSpecMigration` requires `from_graph_version`/`to_graph_version` — `graph_spec.py`, `application_migration.py` → disposition [EVOL_X_N04_GRAPH_VERSION_RECONCILIATION_W1.md](EVOL_X_N04_GRAPH_VERSION_RECONCILIATION_W1.md) |
| **N-05** | Consumers of `rollback_supported` | **VERIFIED (unused)** | Only definition + unit test fixture — no production consumer |
| **N-06** | `RuntimeDeploymentAdapter` implementers | **VERIFIED** | Protocol + `FakeInMemoryRuntimeDeploymentAdapter`, `VenvBundleColocatedRuntimeDeploymentAdapter` — `intergrax/agent_distribution/deployment.py`; production strict paths reject Fake (PROD-Q gates) |
| **N-07** | Baseline pytest collection QUAL-X | **VERIFIED** | 85 tests collected — log `.tmp/session/evol-x-w1/pytest-collect-qual_x.log` |
| **N-08** | CI extras (`dev-ci`) | **VERIFIED (declared)** | `pyproject.toml` `[project.optional-dependencies] dev-ci`; workflow uses `uv sync --extra dev-ci --frozen` pattern (EBH-7 job) |
| **N-09** | `--strict-markers` | **VERIFIED** | `pyproject.toml` `[tool.pytest.ini_options] addopts` includes `--strict-markers` |
| **N-10** | DIST-SCALE-X scope vs EVOL deploy NA | **VERIFIED (parent OPEN)** | P4-A dimensions CLOSED SCOPED; parent DIST-SCALE-X **OPEN** — `DIST_SCALE_X_P4_PHYSICAL_PROVENANCE.md`; NA-03 disposition in OD pack |
| **N-11** | COMPAT-X direction taxonomy | **PARTIAL** | Policy in qualification docs + classifiers; mechanical scan uses write/read direction in `_compat_x_persisted_mechanical_scan.py` — EVOL reuses COMPAT policy via adapter (no new taxonomy) |
| **N-12** | `capability_edge_catalog` | **VERIFIED** | Re-exports `_ALLOWED_EDGES` from `capability_graph.py`; JSON `schema_version` 1.0.0 — align EVOL capability slice edges with this catalog in W2 |
| **N-13** | Ownership ID formats (DUP-X / EBH-3) | **VERIFIED (pattern)** | DUP-X closed-world owner IDs from stage inventories; QUAL-X P3 five owners; EVOL owner rules must cite `source_authority` doc path — W2-03 |
| **N-14** | S8 onboarding path | **VERIFIED** | `intergrax/scaffold/new_application.py`, `package_emit.py`, `cli.py` — S8 fixture may invoke scaffold in harness only (W5) |
| **N-15** | `RuntimeRevision` ↔ git commit | **VERIFIED (no direct field)** | Revision carries `materialization_artifact_digest` etc.; **no** `git_commit_sha` on `RuntimeRevision` — dossier Q5 uses artifact digest + external mapping table (W4-06) |
| **N-16** | Cross-OS digest determinism | **SPEC + W2 gate** | W1: canonical JSON LF policy in master plan §9.6; proof in W2-06 CW-08 on CI ubuntu |
| **N-17** | `reconcile_manifest_integrity()` | **VERIFIED** | Returns `()` on HEAD — executed W1 |
| **N-18** | Full gate unit health | **CLASSIFIED — 32 modules** | [EVOL_X_N18_COLLECTION_CLASSIFICATION_W1.md](EVOL_X_N18_COLLECTION_CLASSIFICATION_W1.md): 8 RC kinds; baseline **INF-TREE** (no START_HEAD checkout replay); **not** W1 regression → **OD-EVOL-10** |
| **N-04** | Graph spec version | **RECONCILED** | See [EVOL_X_N04_GRAPH_VERSION_RECONCILIATION_W1.md](EVOL_X_N04_GRAPH_VERSION_RECONCILIATION_W1.md) — spec inconsistency; not live runtime defect; **N4-A** proposed |
