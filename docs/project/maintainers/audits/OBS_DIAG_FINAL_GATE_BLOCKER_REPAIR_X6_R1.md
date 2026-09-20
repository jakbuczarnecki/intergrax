# OBS-DIAG-X6-R1 — Final Gate Blocker Repair

**Verdict:** `PASS — FINAL GATE BLOCKERS REPAIRED`

Successor repair for `OBS-DIAG-X6 @ 6c31c4840ba45c2ed449ba169a581086f9976bae` (`FAIL`). Does not re-run full X6 or declare enterprise closure.

## Blockers repaired

| Blocker | Root cause | Repair |
| ------- | ---------- | ------ |
| `_field_default` | Refactor moved default resolution to module function; host/scaffold consumers still call `cls._field_default` on `ApplicationSettingsEnvHost` mixins | Restored `_field_default` on `ApplicationSettingsEnvHost`; shared `resolve_field_default` uses dataclass `fields()` / `MISSING` (fail-closed `KeyError`) |
| Qdrant vendor boundary | `client_factory.py` is canonical internal SDK boundary but gate used brittle path suffix list omitting that basename | Integration provider rule: vendor SDK allowed only in named boundary **basenames** under `integrations/providers/**` (includes `client_factory.py`) |
| F401 (12×) | Unused imports in OBS-DIAG qualification tests | Removed unused imports (no `noqa`) |

## Settings contract

```text
canonical owner: ApplicationSettingsEnvHost (+ resolve_field_default helper)
API: cls._field_default(name) -> JsonValue
default handling: field.default when not MISSING
default_factory handling: invokes factory per call
unknown / required-without-default field: KeyError
```

## Qdrant vendor boundary

```text
vendor import owner: intergrax/integrations/providers/vector_store/qdrant/client_factory.py
allowed boundary: INTEGRATION_PROVIDER_VENDOR_BOUNDARY_NAMES under integrations/providers/**
gate rule: maintenance/check_integration_vendor_imports.py semantic basename set
why: opens/data_plane delegate SDK construction to dedicated internal factory; config/rag_store stay vendor-free
```

## Provenance

```text
FAILED_X6_SHA=6c31c4840ba45c2ed449ba169a581086f9976bae
X6_R1_REPAIR_SHA=fb03ae9b4df475e398c40e5aa1398464928913e3
```

## Ready for full X6 re-run

`YES` — blockers addressed; independent re-qualification required.
