# EVOL-X W1 — N-04 graph_version reconciliation

**Baseline SHA:** `5a4e922ffacabef0ee24eb676e5c8705c20ef4a1` (unchanged by W1 docs)

## Facts (verified)

| Artifact | Finding |
|----------|---------|
| `ApplicationGraphSpec` | No `graph_version` field — `intergrax/applications/contracts/graph_spec.py` |
| `GraphSpecMigration` | Requires `from_graph_version` / `to_graph_version` (SemVer strings) — `application_migration.py` |
| `validate_graph_spec_migration` | Validates SemVer + breaking rules — `migration_wiring.py:156–173` |
| `apply_profile_migration` | Applies **profile** migrations only — **no** graph spec apply function in `migration_wiring.py` |
| Tests | Contract tests construct `GraphSpecMigration` with version strings — `test_application_migration_contracts.py` |
| Workflow pinning | Uses `EffectiveProfileRevision` + fingerprint over profile bundle — **not** `graph_version` |

## Disposition

| Question | Answer |
|----------|--------|
| Current **production runtime defect**? | **No** — live graph is `ApplicationGraphSpec` without version axis; no code path reads `graph_version` from spec |
| **Specification inconsistency**? | **Yes** — migration documents reference graph versions that are not stored on the live spec model |
| **Supported migration path affected today**? | **No** — graph migration is **validate-only** at rest; no executor mutates graph from `GraphSpecMigration` |
| **W2 foundational contracts blocked**? | **No** — W2 impact graph can use module/symbol identity; defer graph semver to migration docs |
| **W4 workflow matrix (WM-06/07) blocked**? | **Partially** — matrix rows referencing `GraphSpecMigration` require **OD-EVOL-02 + N-04 disposition** before PASS claims |

## Recommendation [R] (operator — not approved)

**Option N4-A (preferred for W2–W4):** Treat `from_graph_version` / `to_graph_version` as **migration-document identifiers** only; EVOL workflow assessor keys on **EffectiveProfileRevision + fingerprint + pin**, not on absent spec field. Document in capabilities ADR.

**Option N4-B (later Class B):** Add optional `graph_spec_version` to `ApplicationGraphSpec` + wire into fingerprint inputs — **production contract change**; separate OD + domain plan row; **not** W1/W2 without approval.

**Wave placement:** Document N4-A in W1/W2 specs; any N4-B → **authorized production correction** before WM rows claim runtime MIGRATE on graph semver.

## Negative test requirements (W4/W5, not W1)

- Harness with `GraphSpecMigration` whose versions **do not** match any stored spec field → assessor returns **MIGRATE/validate-only** without implying live graph version changed.  
- Attempt in-flight graph semver swap → **EXPLICITLY_UNSUPPORTED** (OD-EVOL-02 policy i).  
- If N4-B approved later: fingerprint change when `graph_spec_version` bumps → must trigger WM-02 not WM-01.

## Production modification

**None** in W1 reconciliation.
