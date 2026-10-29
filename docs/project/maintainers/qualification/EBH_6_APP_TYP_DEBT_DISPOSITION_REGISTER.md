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

## Operator signature (E02-OP-01)

```text
OPERATOR RATIFIED — EBH-6 APP-TYP SCOPED DEBT (EBH6-DEBT-APP-TYP-01..11) — NOT RECORDED
```

## DONE (E02)

**11/11** documented with A/B/C path · operator ratification recorded · auditor **E02-AUD-01** ACCEPT or REJECT with enumerated **B** items.
