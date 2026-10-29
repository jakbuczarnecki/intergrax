# EBH-6 — APP-TYP Debt Disposition Register (E02)

| Field | Value |
| --- | --- |
| **SSOT** | [`_contracts_typing_qualification.py`](../../../tests/qualification/ebh_6/_contracts_typing_qualification.py) |
| **Gate** | [`test_ebh_6_contracts_typing_qualification.py`](../../../tests/qualification/ebh_6/test_ebh_6_contracts_typing_qualification.py) |
| **Operator ratification** | **E02-OP-01 — OPEN** |
| **Independent acceptance** | **E02-AUD-01 — PENDING** |

**Rule:** `no unregistered violations` ≠ zero typing concerns — **11 registered seams** remain on scoped `_shared` wiring surface.

## Disposition key

| Code | Meaning |
| --- | --- |
| **A** | Acceptable scoped debt for EBH-6 — requires operator + auditor ratification |
| **B** | Mandatory fix before Protocol 03 — triggers **E04-APP-*** (**OD-EBH6-01**) |
| **C** | Insufficient evidence — extend qualification |

## 11/11 matrix (@ baseline)

| ID | Symbol | rule_id | Seam type | Proposed | Rationale | Independent |
| --- | --- | --- | --- | --- | --- | --- |
| EBH6-DEBT-APP-TYP-01 | `apply_guardrail_profiles_to_runtime_config` | generic_object | wiring | **A** | Duck-typed RuntimeConfig attachment on wiring path | **PENDING** |
| EBH6-DEBT-APP-TYP-02 | `resolve_harness_host_orchestration_topology_submission_port` | generic_object | harness wiring | **A** | Host port resolver pending typed export | **PENDING** |
| EBH6-DEBT-APP-TYP-03 | `build_scenario_lab_runtime` | semantic_any | diagnostic | **A** | Lab profile builder on scoped surface | **PENDING** |
| EBH6-DEBT-APP-TYP-04 | `build_scenario_production_runtime` | semantic_any | wiring | **A** | Scoped production builder — not contracts layer | **PENDING** |
| EBH6-DEBT-APP-TYP-05 | `certify_business_agents_for_deploy` | semantic_any | wiring | **A** | Factory map — escalate **B** if auditor flags deploy-critical | **PENDING** |
| EBH6-DEBT-APP-TYP-06 | `stable_digest_hex` | semantic_any | helper | **A** | Dynamic payload digest on wiring path | **PENDING** |
| EBH6-DEBT-APP-TYP-07 | `make_scheduler_lifespan` | semantic_any | hosting hook | **A** | Scheduler hook pending hosting contract | **PENDING** |
| EBH6-DEBT-APP-TYP-08 | `resolve_harness_auth_state_from_app` | generic_object | harness | **A** | ASGI handle object until protocol export | **PENDING** |
| EBH6-DEBT-APP-TYP-09 | `apply_profile_migration` | semantic_any | wiring | **A** | Migration entry dynamic profile | **PENDING** |
| EBH6-DEBT-APP-TYP-10 | `resolve_reasoning_task_metadata` | semantic_any | wiring | **A** | Metadata resolver return deferred | **PENDING** |
| EBH6-DEBT-APP-TYP-11 | `attach_recovery_contract_to_task_metadata` | semantic_any | wiring | **A** | Task metadata attachment seam | **PENDING** |

## R1 — debt matching integrity (@ qual HEAD)

| Control | Status |
| --- | --- |
| `_debt_covers` matches `source_path`, `line`, `symbol`, **`rule_id`** | **FIXED** |
| Duplicate `finding_id` rejected | **gate** |
| Wrong `rule_id` cannot mask violation | **negative test** |
| 11 registry rows match live violations | **verified** (0 stale / 0 unregistered) |

## R2 — per-debt qualification (agent recommendation — not operator A)

| ID | Primary evidence | Typing-only risk | Recommendation |
| --- | --- | --- | --- |
| 01 | `_APP_SERVICES_GATE` + registry row | RuntimeConfig duck-type on wiring | **A — SCOPED ACCEPTANCE RECOMMENDED** |
| 02 | same | Harness host port — harness boundary | **A** |
| 03 | same | Lab scenario builder | **A** |
| 04 | same | Prod scenario builder — **review deploy adjacency** | **A** (auditor may → **B**) |
| 05 | same | Deploy certification factories — **authority-sensitive** | **A** (auditor may → **B**) |
| 06 | same | Digest helper | **A** |
| 07 | same | Scheduler lifespan hook | **A** |
| 08 | same | Harness auth ASGI handle | **A** |
| 09 | same | Profile migration entry | **A** |
| 10 | same | Reasoning metadata resolver | **A** |
| 11 | same | Recovery contract on metadata — **recovery semantics** | **A** (runtime validation separate from typing) |

Runtime fail-closed/admission for 04/05/11 not proven by typing gate alone — documented as scoped debt, not invariant waiver.


| ID | Production reachability | Public boundary | Trust/authority | Pos gate | Neg gate | Proposal |
| --- | --- | --- | --- | --- | --- | --- |
| 01 | Tier-3 wiring imports | `_shared` wiring | Guardrail profile attach — not tenant auth | seam gate PASS | unregistered/stale debt tests | **A** |
| 02 | Harness host wiring | harness-only resolver | Host topology port — lab/harness | seam gate PASS | same | **A** |
| 03 | Scenario lab path | diagnostic builder | No deploy authority | seam gate PASS | same | **A** |
| 04 | Production scenario wiring | `_shared` builder | **Special audit:** scenario prod runtime — monitor deploy adjacency | seam gate PASS | same | **A** (auditor may → **B**) |
| 05 | Deploy certification wiring | `_shared` factory map | **Special audit:** deploy certification — authority-sensitive | seam gate PASS | same | **A** (auditor may → **B**) |
| 06 | Digest helper on wiring | helper | Low authority | seam gate PASS | same | **A** |
| 07 | FastAPI lifespan hook | hosting | Scheduler lifecycle — not execution identity | seam gate PASS | same | **A** |
| 08 | Harness auth resolver | harness ASGI handle | Auth state — harness boundary | seam gate PASS | same | **A** |
| 09 | Profile migration entry | wiring | Migration — not governance | seam gate PASS | same | **A** |
| 10 | Reasoning metadata | wiring | Metadata resolver | seam gate PASS | same | **A** |
| 11 | Recovery metadata attach | wiring | **Special audit:** recovery contract on task metadata | seam gate PASS | same | **A** |

Positive: `test_ebh6_application_services_production_seam_typing_passes` · Negative: `test_ebh6_application_services_typing_negative_unregistered_debt_fails`, `test_ebh6_application_services_typing_negative_stale_debt_detected`, contracts typing negative gates.

## C2-04 — Operator handoff (E02-OP-01 — proposed, NOT APPROVED)

```text
OPERATOR RATIFIED — EBH-6 APP-TYP SCOPED DEBT (EBH6-DEBT-APP-TYP-01..11 category A)
Conditions: no new unregistered seams on scoped _shared surface; auditor E02-AUD-01 ACCEPT;
items 04/05 escalated to B if deploy/admission authority impact rejected.
```


## DONE (E02)

**11/11** documented with A/B/C path · operator ratification recorded · auditor **E02-AUD-01** ACCEPT or REJECT with enumerated **B** items.
