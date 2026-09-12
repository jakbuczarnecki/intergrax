<!--
© Artur Czarnecki. All rights reserved.
Intergrax is source-available under the Intergrax Evaluation and Collaboration License 1.0.
-->

# ERL admission boundary

**Purpose:** Provide a single, domain-neutral entry from an external effect occurrence into the Enterprise Reliability Layer (ERL), linking `ExternalEffectOutcome` truth to reliability case lifecycle initialization without duplicating reconciliation, evidence, resolution, or recovery ownership.

**Primary modules:**

| Layer | Module |
| ----- | ------ |
| Contracts | `intergrax/contracts/enterprise_reliability/admission_boundary.py` |
| Runtime | `intergrax/runtime/enterprise_reliability/admission_boundary.py` |

---

## Position in ERL

```text
External effect (outcome + refs)
        │
        ▼
ERL admission boundary  ← this document
        │
        ├── classification (reliability projection)
        ├── UNKNOWN uncertainty admission (existing contract admission)
        └── reliability case lifecycle init + handoff
        │
        ▼
Existing orchestration (reconciliation, evidence, resolution, …)
```

The boundary **reuses** existing types:

- `ExternalEffectOutcome` and `project_external_effect_to_reliability`
- `admit_external_effect_unknown_with_contract` (`contract_admission.py`)
- `initial_reliability_case_lifecycle` and `transition_reliability_case_lifecycle`

It does **not** introduce payment, order, or customer fields, and does not add registries or singleton handlers.

---

## Ownership

| Owns | Does not own |
| ---- | ------------ |
| Validating admission requests | Reconciliation decisions |
| Classifying outcome for Reliability interaction | Evidence gathering |
| Opening UNKNOWN episodes with contract posture | Resolution or compensation execution |
| Creating reliability case records and handing off to `RECONCILIATION_RUNNING` | Recovery execution or governance |

---

## Admission pipeline (not a second case lifecycle)

Admission uses a short **pipeline phase** enum (`ExternalEffectAdmissionPhase`) distinct from `ReliabilityCaseLifecycleState`:

1. **RECEIVED** — request validated (`assert_external_effect_admission_request`, effect contract validation).
2. **CLASSIFIED** — `project_external_effect_to_reliability` determines whether ERL case work is required.
3. **CASE_CREATED** — for `UNCERTAINTY_FAIL_CLOSED`, uncertainty state and `initial_reliability_case_lifecycle` run.
4. **HANDED_OFF** — case transitions to `RECONCILIATION_RUNNING`; result returned to callers for downstream orchestration.

Definitive `SUCCESS` or `FAILURE` outcomes complete at **HANDED_OFF** with projection only (no reliability case), preserving fail-closed Reliability semantics without opening an uncertainty episode.

---

## Correlation and references

Admission requests carry:

- `external_effect_ref` — opaque handle to the external effect occurrence
- `correlation_id` — stable chain across effect, case, and orchestration
- optional `uncertainty_state_ref` — must match `erl:uncertainty:{correlation_id}` when supplied

Platform helpers:

- `uncertainty_state_ref_for_correlation`
- `reliability_case_id_for_admission`

The admission **result** binds `external_effect_ref`, `correlation_id`, and optional `case_record` so integrators can persist the link without domain payloads in core contracts.

---

## Failure model

Explicit errors (no silent continue):

- `ExternalEffectAdmissionContextError` — invalid or inconsistent request (e.g. missing correlation, wrong uncertainty ref).
- `ExternalEffectAdmissionCaseError` — reliability case initialization failed after UNKNOWN admission.

---

## Relation to other ERL hubs

| Hub | Relationship |
| ----- | ------------ |
| [`UNCERTAINTY_MANAGEMENT.md`](UNCERTAINTY_MANAGEMENT.md) | UNKNOWN lifecycle after contract-aware admission |
| [`EXTERNAL_EFFECT_CONTRACTS.md`](EXTERNAL_EFFECT_CONTRACTS.md) | Required `ExternalEffectContract` on admission |
| [`RECONCILIATION.md`](RECONCILIATION.md) | Next step after handoff (`plan_external_effect_reconciliation`, …) |
| [`ENTERPRISE_RELIABILITY_LAYER.md`](ENTERPRISE_RELIABILITY_LAYER.md) | Parent capability map |

---

## Domain neutrality

Admission only understands canonical outcome, contract safety declarations, correlation, and opaque refs. Domain truth (ledger rows, carrier labels, PSP codes) remains in plugins, applications, or scenario stores; ERL receives **that an effect is uncertain** and **where to continue** in the shared case lifecycle.
