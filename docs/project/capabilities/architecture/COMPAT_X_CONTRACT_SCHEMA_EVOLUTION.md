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
| Plugin / provider compatibility policy (cross-platform rules) | This document — **Plugin/Provider Compatibility Policy** (below) |
| Package ↔ Intergrax platform version compatibility | `intergrax/core/distribution/platform_compatibility.py` |
| Plugin package compatibility declaration | `PlatformPluginManifest.platform_compatibility` in `intergrax/core/plugins/package_contract.py` |
| Provider contract identity / registration | `IntegrationContractSpec` + canonical `intergrax/integrations/registry/catalog.py` |
| External provider contract semantic assessment | `intergrax/integrations/contracts/external_contract_compatibility.py` (+ extensions/service composition) |

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

## Event Evolution Policy

RuntimeEvent evolution has **two independent compatibility dimensions**. They must not be conflated into a single generic version string or a second platform-wide event version registry.

| Dimension | Identity field | Canonical current-value owner | Enforcement |
| --- | --- | --- | --- |
| **Envelope** | `RuntimeEvent.schema_version` (e.g. `runtime_event.v2`) | **`intergrax/contracts/runtime_event.py`** (`RUNTIME_EVENT_SCHEMA_VERSION`); `RUNTIME_SCHEMA_REGISTRY["runtime_event"]` is a derived catalog projection | `intergrax/runtime/events/schema_guard.py`, `parse_runtime_event_payload` |
| **Payload** | `payload.payload_schema_id` + registered `RuntimeEventPayload` class | Payload models under `intergrax/runtime/events/payloads/` + `intergrax/runtime/events/payload_registry.py` | `schema_guard`, `validate_payload_envelope`, canonical write policy |

### Envelope version semantics

- **Current write and supported read** for production runtime evidence is `RUNTIME_EVENT_SCHEMA_VERSION` on the contract (registry entry must match; preview hooks do not apply to the `runtime_event` envelope).
- **Missing** envelope `schema_version` on persisted/wire decode ⇒ **fail closed** (`parse_runtime_event_payload`).
- **Unknown or future** envelope versions ⇒ **fail closed** (no implicit preview acceptance via qualification).
- **Retired** envelope versions (e.g. `runtime_event.v1`) ⇒ **fail closed** on read; no default v1→v2 migration or compatibility adapter in event evolution policy.
- `PREVIEW_RUNTIME_SCHEMA_VERSIONS` is a conformance preview hook only; it is **not** an evolution policy authority.

### Payload version semantics

- Each registered canonical payload is a distinct **schema identity** (`payload_schema_id`); version lineage is evidenced by explicit registration of multiple classes, not by parsing the ID string.
- **Supported read**: any canonical payload class registered in `CANONICAL_PAYLOAD_TYPES` / `payload_registry` may be parsed when referenced by `payload_schema_id` (subject to structural validation).
- **Current canonical write**: for each `RuntimeEventType`, the preferred `payload_schema_id` is owned solely by `intergrax/runtime/events/runtime_event_payload_policy.py` (`iter_runtime_event_payload_policies` / `EVENT_TYPE_PREFERRED_SCHEMA`). Write-time enforcement requires an exact match for strict spine production types.
- **Unknown** `payload_schema_id` ⇒ **fail closed** (`UnknownPayloadSchemaError`).
- Historical payload versions remain **factual evidence** when read; they do **not** imply reusable current Execution or Governance authority.

### Extension / domain signal evolution

Extension payloads and `DOMAIN_SIGNAL` / custom `event_kind` surfaces obey:

1. Platform canonical registry IDs cannot be overridden by extension registration.
2. Duplicate `schema_id` bound to a different class fails closed at registration.
3. Unknown extension schema IDs fail closed on validation.
4. Evolution requires a **new explicit schema identity**; no implicit mutation of an existing ID.
5. The platform does not infer backward compatibility for extension schemas.
6. Extension registration does not grant Execution or Governance authority.
7. Event compatibility never widens tenant scope.

Closed-world enumeration of third-party schemas is out of scope; the **extension contract** above is normative.

### Ownership split (exactly one concern owner each)

```text
COMPAT-X architecture (this document)
  → cross-platform event evolution rules and dispositions

RuntimeEvent contract + RUNTIME_SCHEMA_REGISTRY
  → current envelope version truth

Payload models + payload_registry
  → payload schema identities and registered readable types

runtime_event_payload_policy
  → canonical write selection per RuntimeEventType

event_kind_registry
  → extension event_kind composition binding

schema_guard + ValidatingRuntimeEventPersistence (production wiring)
  → enforcement on canonical production write boundaries

Qualification inventory (tests/qualification/compat_x/)
  → mechanical evidence only; NOT semantic authority
```

Qualification code must mirror policy and detect drift; it must **not** become a second registry of current production versions.

## Plugin / Provider Compatibility Policy (COMPAT-X-R4)

Cross-platform rules for plugin packages and integration providers. **COMPAT-X** owns policy coordination only — not runtime evaluation, registration, or activation.

### Policy rule 1 — one cross-platform policy owner

This architecture document is the **semantic owner** of cross-platform plugin/provider compatibility and evolution rules. Qualification under `tests/qualification/compat_x/` mirrors and gates; it is not semantic authority.

### Policy rule 2 — concrete domain semantics remain domain-owned

```text
package ↔ Intergrax platform version compatibility
  → intergrax/core/distribution/platform_compatibility.py

plugin package declaration (platform_compatibility metadata)
  → PlatformPluginManifest in intergrax/core/plugins/package_contract.py

provider registration / contract identity
  → Integration Catalog (catalog.py) + IntegrationContractSpec (contract_spec.py)

external provider contract semantic assessment
  → intergrax/integrations/contracts/external_contract_compatibility.py
     (+ external_contract_compatibility_extensions.py / service composition)
```

`intergrax/runtime/integrations/registry_v2.py` is a **derived projection/read model** from canonical catalog truth. It is not a second registration authority, discovery universe, or compatibility authority.

### Policy rule 3 — compatibility does not grant authority

```text
compatible ≠ installed ≠ configured ≠ effective ≠ qualified ≠ authorized ≠ executable
```

A compatible outcome may be **required evidence** for admission or qualification; it cannot mint Governance permission, Execution authority, or activation by itself.

### Policy rule 4 — fail closed

Unknown, incomplete, or ambiguous compatibility information must never silently become “compatible”. Missing external evidence, unsupported or ambiguous evaluators, stale evidence, tenant/provider/operation mismatch, and evidence conflict must not yield `COMPATIBLE`.

### Policy rule 5 — no second mechanism

No domain may recreate an equivalent platform-level package/platform compatibility evaluator when `platform_compatibility.py` already defines canonical semantics. Wrappers (e.g. `check_manifest_platform_compatibility`) must delegate without independent version-range logic.

### Policy rule 6 — extensibility

External plugin/provider compatibility extensions enter through **typed** platform/domain contracts (`ExternalContractCompatibility*` family) and remain structurally replaceable without patching core.

### Hard separation (do not unify)

```text
package ↔ platform compatibility
≠ provider registration / contract identity
≠ external contract semantic compatibility
≠ production qualification
≠ configuration selection
≠ Governance permission
≠ Execution authority
```

## Relationship to certification

Mechanical inventory, gates, and FRZ-CMP candidate evidence: [`docs/project/maintainers/qualification/COMPAT_X_CONTRACT_SCHEMA_EVOLUTION_CERTIFICATION.md`](../../maintainers/qualification/COMPAT_X_CONTRACT_SCHEMA_EVOLUTION_CERTIFICATION.md).

Program plan: [`docs/project/maintainers/plans/PLATFORM_ENTERPRISE_COMPLETION_ROADMAP.md`](../../maintainers/plans/PLATFORM_ENTERPRISE_COMPLETION_ROADMAP.md) (**COMPAT-X**).
