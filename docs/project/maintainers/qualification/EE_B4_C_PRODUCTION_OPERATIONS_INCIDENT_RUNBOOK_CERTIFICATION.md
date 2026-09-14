# EE-B4-C — Production Operations, Incident & Runbook Certification

**Task:** EE-B4-C  
**Branch:** `development`

## Provenance

| Field | Value |
|-------|-------|
| **START_HEAD** | `6baa9b4fa92addfad4e163330b3a7e75100b3629` |
| **START_ORIGIN** | `6baa9b4fa92addfad4e163330b3a7e75100b3629` |

## Deliverables

| Artifact | Path |
| -------- | ---- |
| Incident model | `docs/project/maintainers/architecture/EXECUTION_ENGINE_PRODUCTION_OPERATIONS_INCIDENT_MODEL.md` |
| Runbooks | `docs/project/maintainers/runbooks/EXECUTION_ENGINE_PRODUCTION_RUNBOOKS.md` |
| Validation support | `testing_support/operations/` |
| Gate tests | `tests/unit/runtime/architecture/test_ee_b4_c_*.py` |

**PRODUCTION CODE CHANGED:** NO (`intergrax/` unchanged).

## Incident matrix

| Incident | Scope | Severity | Readiness impact | Canonical owner | Runbook |
| -------- | ----- | -------- | ---------------- | --------------- | ------- |
| INC-01 | Global root capacity | SEV-2 typical | not_ready on REJECT | EE-B1.2 capacity admission | RB-02 |
| INC-02 | Mandatory persistence | SEV-1 | not_ready fail closed | Runtime event / mandatory store | RB-03 |
| INC-03 | Export plane | SEV-4 | ready may hold | Observability export boundary | RB-04 |
| INC-04 | Worker pool | SEV-3 | global often ready | EE-B1.3 containment | RB-06 |
| INC-05 | Nexus fan-out | SEV-3 | orchestration scoped | Nexus orchestration | RB-06 + model §8 |
| INC-06 | Provider dependency | SEV-3 | profile-dependent | Tool/integration plane | RB-09 |
| INC-07 | Per execution retry | SEV-3 | execution terminal | NPSC-5E recovery | RB-10 |
| INC-08 | Recovery plane | SEV-2 | scoped block | NPSC-5E recovery | RB-07 |
| INC-09 | Checkpoint store | SEV-2 | recovery ineligible | Checkpoint + recovery admission | RB-08 |
| INC-10 | Governance admission | SEV-3 | not unhealthy by default | Governance admission | RB-12 |
| INC-11 | Process shutdown | SEV-2 | not_ready | EE-B4-B shutdown | RB-13, RB-05 |
| INC-12 | Runtime fatal | SEV-1 | not_ready | Execution runtime host | RB-01 |
| INC-13 | Single tenant/profile | SEV-3 | tenant scoped | Tenant-scoped facts | RB-14 |
| INC-14 | Unclassified | SEV-2 | classify first | Operator → taxonomy | RB-01 |

## Safe / unsafe action matrix

| Situation | Safe action | Forbidden action |
| --------- | ----------- | ---------------- |
| Capacity saturated | Capacity/readiness diagnosis; wait for terminal drain | Manual permit/capacity counter manipulation |
| Mandatory evidence down | FAIL CLOSED; restore store | Disable mandatory persistence; delete events |
| OTLP export down | DEGRADED; keep execution | Stop engine for telemetry alone |
| Unknown side effect | NO BLIND RETRY; reconcile | Manual provider/tool rerun |
| Recovery failed | Read checkpoint; NPSC-5E path | Edit/delete checkpoint JSON |
| Governance DENY spike | Triage policy/client/attack | Disable governance; force allow |
| Shutdown stuck | EE-B4-B phased diagnosis | Skip flush/persist ordering |
| Tenant-only errors | Tenant-scoped runbook | Global restart without proof |
| Retry exhausted | Classify terminal vs unknown | Hidden/manual retry chain |
| Provider outage | Adapter/deployment appendix | Direct provider invocation |

## Architecture gates

| Check | Expected |
| ----- | -------- |
| IncidentRuntime / RunbookEngine / OperatorRuntime | 0 in `intergrax/` |
| Second recovery/retry/evidence/governance/execution owner | 0 |
| NPSC-5F protected surfaces modified | NO |

**NPSC-5F HANDOFF:** NONE

## Tests

| Module | Focus |
| ------ | ----- |
| `test_ee_b4_c_incident_taxonomy.py` | INC catalog vs model |
| `test_ee_b4_c_runbook_completeness.py` | RB-01–14 sections |
| `test_ee_b4_c_forbidden_operator_bypass.py` | Action-section forbidden scan |
| `test_ee_b4_c_capacity_runbook.py` | Diagnosis not permit hack |
| `test_ee_b4_c_evidence_runbook.py` | FAIL CLOSED vs DEGRADED |
| `test_ee_b4_c_recovery_runbook.py` | No checkpoint edit guidance |
| `test_ee_b4_c_unknown_side_effect_runbook.py` | NO BLIND RETRY |
| `test_ee_b4_c_shutdown_runbook.py` | EE-B4-B linkage |
| `test_ee_b4_c_provider_neutrality.py` | Core RB vendor-free |
| `test_ee_b4_c_operations_architecture_gate.py` | Docs + forbidden runtimes |

## Regression matrix

EE-B4-A, EE-B4-B, EE-B3-A/C architecture gates, EE-B2 architecture gate, EE-B1 relevant gates, NPSC-5E recovery slice, NPSC-5F sentinels (if green on HEAD) — session pytest log.

## Final verdict

**PASS** when B4-C suite + regression + static checks complete with zero forbidden bypass guidance in operator action sections.
