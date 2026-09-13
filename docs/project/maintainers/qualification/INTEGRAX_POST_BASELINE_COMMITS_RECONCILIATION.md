# INTEGRAx-POST-BASELINE-COMMITS-RECONCILIATION

## Metadata

| Field | Value |
| ----- | ----- |
| **Task** | `INTEGRAx-POST-BASELINE-COMMITS-RECONCILIATION` |
| **Date** | 2026-09-13 |
| **Branch** | `development` |
| **Audit range** | `009ad0c62ba6afbd07a7e4f2dbe8b4bbbab1d2c6..18bded87b0fc311ee9d9b76f9223383e0ec6b2c1` |
| **Prior eligible baseline (configuration contract)** | `009ad0c62ba6afbd07a7e4f2dbe8b4bbbab1d2c6` |
| **Certified code baseline candidate (post-audit)** | `118798759e8a198b9a1d21ecd93293fe601fd7d9` |
| **Certification / evidence HEAD** | `18bded87b0fc311ee9d9b76f9223383e0ec6b2c1` |

**Chain status:** **CHAIN ACCEPTED WITH OBSERVATIONS**

Formal core-platform **freeze was not executed** in this task. Working tree remained dirty (Execution Engine reliability WIP); see [`INTEGRAX_CORE_PLATFORM_FREEZE.md`](INTEGRAX_CORE_PLATFORM_FREEZE.md).

---

## Commit inventory

| SHA | Message | Type | Class | Verdict |
| --- | ------- | ---- | ----- | ------- |
| `118798759…` | refactor(tracing): harden public trace value contracts | Production (`intergrax/contracts/**`) | **B** | **ACCEPT WITH OBSERVATIONS** |
| `2eb7cb463…` | INTEGRAx-CORE-PLATFORM-FREEZE | Documentation only | **DOCS-ONLY** | **ACCEPT WITH OBSERVATIONS** |
| `8523b1574…` | fix(certification): repair npsc5f frozen baseline provenance | Qualification / tests | **EVIDENCE** | **ACCEPT** |
| `888584b98…` | INTEGRAx-PRE-FREEZE-BASELINE-RECONCILIATION | Documentation only | **DOCS-ONLY** | **ACCEPT** |
| `18bded87b…` | INTEGRAX-PLUGIN-CONFIGURATION-CONTRACT-SCOPED-RECERTIFICATION | Qualification / tests | **EVIDENCE** | **ACCEPT** |

---

## Tracing commit (`118798759…`) — summary

| Criterion | Result |
| --------- | ------ |
| Contract impact | Typed `TraceObject` / `StructuredJsonValue`; stricter JSON-safe validation on trace payloads and tags |
| Identity impact | **None** — `run_id` remains correlation field on `TraceEvent`; `TraceEvent.new_id()` mints **event_id** only |
| Tracing ownership | Plane B read-model contracts; runtime re-exports same canonical `TraceEvent` class (contract test) |
| Vendor abstraction | No OTLP/vendor coupling in diff; validation layer only |
| Runtime impact | No Execution Engine / authorization / lifecycle ownership changes |
| Classification | **CLASS B** |
| Verdict | **ACCEPT WITH OBSERVATIONS** (public typing surface tightened; no scoped re-certification required) |

---

## NPSC5F provenance commit (`8523b1574…`) — summary

| Criterion | Result |
| --------- | ------ |
| Provenance contract | Git ancestry guards in `testing_support/frozen_baseline_provenance.py` (qualification-only) |
| Evidence ownership | Repairs frozen baseline SHA reachability checks for NPSC5F gates |
| Identity impact | **None** |
| Persistence abstraction | **N/A** — no domain persistence imports |
| Runtime control impact | **None** — qualification harness only |
| Classification | **EVIDENCE** (not production) |
| Verdict | **ACCEPT** |

---

## Scoped re-certification

```text
Requires scoped recertification: NO
```

Scoped re-certification for plugin configuration contract at `009ad0c62…` is recorded in commit `18bded87b…`. Tracing hardening classified **CLASS B** with targeted contract tests passing.

---

## Verification (2026-09-13)

| Check | Result |
| ----- | ------ |
| Targeted pytest slice (58 tests) | **PASS** |
| `ruff check` (tracing contracts) | **PASS** |
| `ruff format --check` (tracing contracts) | **FAIL** — `__init__.py`, `values.py` would reformat (pre-existing; no mass cleanup in this task) |
| `pyright` (tracing contracts) | **PASS** |

---

## Findings

| Severity | Finding |
| -------- | ------- |
| **Major** (historical) | `2eb7cb463…` asserted **FROZEN** in SSOT; superseded by `888584b98…` (**FREEZE PREPARED**). |
| **Minor** | Ruff format drift on two tracing contract files at audited HEAD. |
| **Observation** | Uncommitted WIP (execution reliability, compensation wiring) **excluded** from baseline audit. |

---

## Operator note

Committed history under audit **≠** uncommitted working tree. Formal freeze remains **out of scope** until clean-tree verification.
