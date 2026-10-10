# COMPAT-X — Cross-Platform Contract, Schema & Evolution Policy

**Status:** Canonical **semantic** policy authority for COMPAT-X (qualification mirrors this document; runtime behavior unchanged).

**Production delta:** `0` (architecture + qualification enforcement only).

## Authority split (normative)

```text
COMPAT-X owns cross-platform evolution rules.

Domain contract/schema owners own concrete current version values.

Qualification code mechanically mirrors and enforces this policy.

Qualification code is not semantic authority.
```

| Role | Canonical artifact |
| --- | --- |
| Cross-platform semantic policy | This document (`docs/project/capabilities/architecture/COMPAT_X_CONTRACT_SCHEMA_EVOLUTION.md`) |
| Executable qualification enforcement | `tests/qualification/compat_x/` (classification, gates, adversarial probes) |
| Public contract version truth | `intergrax/contracts/migrations/registry.py` (`CONTRACT_SCHEMA_REGISTRY`) |
| Runtime persisted/wire schema truth | `intergrax/runtime/schema/registry.py` (`RUNTIME_SCHEMA_REGISTRY`) |
| Event payload schema truth | `intergrax/runtime/events/payload_registry.py` |
| Plugin / provider manifest compatibility | Domain owners per owner matrix (see qualification `_compat_x_owner_discovery.py`) |

There is **no** universal platform version registry. Registries remain complementary domain authorities.

## Evolution rules

1. For a compatibility-relevant surface, the **owning domain** controls version identity; COMPAT-X defines **how** evolution is classified and gated.
2. Breaking semantic or structural change ⇒ **new** contract/schema version identity (bump required).
3. Unknown compatibility impact ⇒ **fail closed** until explicitly classified.
4. Old-version acceptance ⇒ only through explicit reader, migration, or compatibility adapter policy — never implicit.
5. **Additive** backward compatibility ⇒ permitted only when an explicit **family** compatibility policy and evidence say so (not inferred from Pydantic defaults alone).
6. **Compatibility adapters** (`intergrax/compat/…`) are translation/read-compat layers; they do **not** hold canonical current-version authority unless version identity is explicitly inherited from a canonical versioned envelope owner (never the shim module itself).

## Version obligation vs compliance

**Obligation** states what a surface must satisfy (e.g. `EXPLICIT_VERSION_REQUIRED`).

**Compliance** states whether current HEAD satisfies that obligation (e.g. `VERSION_PRESENT_AND_OWNED` vs `VERSION_REQUIRED_BUT_MISSING`).

```text
EXPLICIT_VERSION_REQUIRED + missing/unknown current version ≠ COMPLIANT
```

Persisted-schema defects discovered in P0 remain **R2** remediation; R1 must detect and classify them accurately without suppressing findings.

## Typed authority disposition

Canonical current-version authority for a surface is one of:

- `DOMAIN_VERSION_OWNER` — domain registry or module owns current version value.
- `INHERITED_VERSION_OWNER` — version owned by enclosing canonical envelope (not the adapter).
- `NO_VERSION_AUTHORITY` — no canonical current-version owner (internal surfaces, compatibility adapters).

Optional owner path is present only when disposition is `DOMAIN_VERSION_OWNER` or `INHERITED_VERSION_OWNER`.

## Relationship to certification

Mechanical inventory, gates, and FRZ-CMP candidate evidence: [`docs/project/maintainers/qualification/COMPAT_X_CONTRACT_SCHEMA_EVOLUTION_CERTIFICATION.md`](../../maintainers/qualification/COMPAT_X_CONTRACT_SCHEMA_EVOLUTION_CERTIFICATION.md).

Program plan: [`docs/project/maintainers/plans/PLATFORM_ENTERPRISE_COMPLETION_ROADMAP.md`](../../maintainers/plans/PLATFORM_ENTERPRISE_COMPLETION_ROADMAP.md) (**COMPAT-X**).
