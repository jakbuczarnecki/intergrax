# GR-12 Final Control-Plane Qualification

**Status:** `READY FOR AUDIT`  
**Task:** GR-12-FINAL (parent qualification under GOV-X1)  
**AUDITED/IMPLEMENTED HEAD (session baseline):** `b706c2c72a900575ec360b7f217c97a3656c71b9` on `development`  
**SSOT inventory:** `tests/qualification/governance/gr12/catalog.py` (`GR12_CONTROL_PLANE_SURFACES`)  
**Parent mechanical gate:** `tests/qualification/governance/gr12/test_gr12_final_control_plane_qualification.py`  
**GOV_FINAL_4 scenario CP:** remains **GAP** (honesty gate); **candidate** status `READY FOR AUDIT` per `GR12_GOV_FINAL_4_CP_QUALIFICATION_CANDIDATE_STATUS`.

Independent exact-SHA acceptance is required before `GR-12 = CLOSED` or uplifting G3B **CONTROL_PLANE_MUTATION** to **COVERED**.

---

## Canonical authority graph

```text
RequestIdentity
        ↓
domain/application control-plane orchestration
        ↓
typed ControlPlaneMutationRequest
        ↓
ControlPlaneMutationAuthorizationBoundary
        ↓
ControlPlaneMutationPolicyEvaluator
        ↓
PolicyDecision (ALLOW / DENY / REQUIRE_HUMAN / ESCALATE)
        ↓
ControlPlaneMutationAuthorizationEvidence
        ↓
domain mutation owner (AD, AHI, ECP, task control, catalog registry, vector admin port, …)
```

**Exception:** current Memory writes — execution/background domain authority via `MemorySecurityGovernanceService` (**GR-12 NOT_APPLICABLE** for `CP-MEM-SPECIALIZED-MUTATION`).

---

## Closed-world surface inventory

Authoritative fields: `path_id`, `surface`, `production_entrypoint`, `mutation`, `consequential`, `applicability`, `coverage`, identity/scope (`current_authority`), permission (`current_guard` + CLA-04 boundary), mutation owner (`recommended_owner`), evidence (`audit_evidence`), proof (`qualification_proof`).

| path_id | Applicability | Coverage | Qualification proof |
| --- | --- | --- | --- |
| CP-AD-ACTIVATE … CP-AD-POST-CUTOVER-FAIL (11) | APPLICABLE | QUALIFIED | unit AD governance / remediation tests |
| CP-AHI-APPLY, CP-AHI-ROLLBACK | APPLICABLE | QUALIFIED | `test_ahi_control_plane_governance.py` |
| CP-ECP-SCALE-K8S, CP-ECP-SCALE-CELERY | APPLICABLE | QUALIFIED | `test_ecp_control_plane_governance.py` |
| CP-TASK-CANCEL, CP-TASK-RESUME, CP-TASK-AUTONOMY | APPLICABLE | QUALIFIED | task control governed tests |
| CP-REF-ACTIVATE | APPLICABLE | QUALIFIED | AD activation remediation |
| CP-HOST-BOUNDARY-OPTIONAL, CP-ECP-BOUNDARY-OPTIONAL | APPLICABLE | QUALIFIED | A2 composition + ECP wiring |
| CP-PLUGIN-CATALOG-HOT-RELOAD | APPLICABLE | QUALIFIED | GR-12-A4-R1-R1 qualification module |
| CP-VECTOR-INDEX-ADMIN | APPLICABLE | QUALIFIED | GR-12-A4-R2-R1 qualification module |
| CP-MEM-SPECIALIZED-MUTATION | NOT_APPLICABLE | NOT_APPLICABLE | R3-R1 memory qualification (no CLA-04 on execution writes) |
| CP-MARKETPLACE-ACQUIRE | NOT_APPLICABLE | NOT_APPLICABLE | recommendation/handoff only |
| CP-BOOT-PLUGIN-REGISTER | NOT_APPLICABLE | NOT_APPLICABLE | process-start registry |
| CP-POLICY-BUNDLE-COMPOSE | NOT_APPLICABLE | NOT_APPLICABLE | compose-time bundle only |

Full row detail (entrypoints, guards, owners): see `GR12_CONTROL_PLANE_SURFACES` in catalog SSOT.

---

## NOT_APPLICABLE matrix (evidence)

| path_id | Why not live CP mutation | Actual authority | Revisit trigger |
| --- | --- | --- | --- |
| CP-MEM-SPECIALIZED-MUTATION | Execution/background memory writes | `MemorySecurityGovernanceService` | Live operator memory admin API |
| CP-MARKETPLACE-ACQUIRE | No install mutation in service | Catalog governance evaluator | Live install authority |
| CP-BOOT-PLUGIN-REGISTER | Startup registry population | Host startup / profile validation | Hot plugin admission API |
| CP-POLICY-BUNDLE-COMPOSE | Immutable compose-time bundle | Environment profile | Live policy activation API |

---

## Ownership / composition

| Concern | Owner |
| --- | --- |
| CLA-04 contracts | `intergrax/contracts/control_plane_mutation.py` |
| Authorization boundary | `intergrax/runtime/governance/control_plane_mutation_authorization.py` |
| Policy evaluator | Injectable `ControlPlaneMutationPolicyEvaluator` |
| Operator orchestration | Application/domain services |
| Mutation execution | Domain owners (AD, registry, vector port, ECP executor, …) |
| Provider mechanics | Integration/provider adapters (no business permission) |
| Evidence | `ControlPlaneMutationAuthorizationEvidence` contract |

Mandatory composition: `CP-HOST-BOUNDARY-OPTIONAL`, `CP-ECP-BOUNDARY-OPTIONAL` — omission fails closed (A2/A3 composition proofs).

---

## Bypass scan (session)

| Candidate | Classification |
| --- | --- |
| `reload_integration_catalog` outside governed service | **FALSE POSITIVE** if absent — CHR-13 AST inventory empty |
| Live `register_integration` outside bootstrap + governed reload | **IN CATALOG / QUALIFIED** — CHR-16 |
| AD/AHI/ECP/Task/Vector operator paths | **IN CATALOG / QUALIFIED** |
| Memory execution writes | **N/A WITH EVIDENCE** — specialized governance, not CLA-04 CP |
| UAEP / MSE / inference | **FALSE POSITIVE** — `GR12_EXECUTION_PLANE_EXCLUSIONS` |

Structural regression: `GR12_FINAL_BYPASS_REGRESSION_PROOF_NODES` + catalog closed-world gate F24.

---

## Fresh authorization / stale / HITL

| Mechanism | Evidence |
| --- | --- |
| Tenant/scope mismatch | `GR12_FINAL_TENANT_SCOPE_NEGATIVE_PROOF_NODES` |
| Stale revision / CAS | `GR12_FINAL_STALE_REVISION_NEGATIVE_PROOF_NODES` |
| HITL ≠ ALLOW | `GR12_FINAL_HITL_NEGATIVE_PROOF_NODES` |
| Authorization evidence | `GR12_FINAL_EVIDENCE_PROOF_NODES` |
| External evaluator pluginability | `GR12_FINAL_PLUGINABILITY_PROOF_NODES` |

---

## Weak-boundary scan (GR-12 seams)

Production CLA-04 contracts use Pydantic models with `extra="forbid"`; boundary module is typed. Qualification tests retain **TRACKED FREEZE DEBT** `# type: ignore[assignment]` in memory R3-R1 only — not expanded in this task.

---

## Tests

| Command | Purpose |
| --- | --- |
| `uv run --frozen pytest tests/qualification/governance/gr12/test_gr12_final_control_plane_qualification.py -q -p no:xdist` | Parent gate F1–F25 |
| `uv run --frozen pytest tests/qualification/governance/gr12/ -q -p no:xdist` | Full GR-12 suite (double replay) |
| `GR12_FINAL_REPRESENTATIVE_EXECUTION_PROOF_NODES` | Representative production proof batch |

---

## FRZ mapping (scoped evidence only — global status remains OPEN)

| FRZ | GR-12-FINAL contribution |
| --- | --- |
| FRZ-GOV-01..10 | Scoped control-plane permission vs execution separation; fail-closed composition; fresh auth on applicable paths |
| FRZ-TRC-01, 06, 07, 08 | Partial — authorization evidence on qualified paths; full trace closure → GR-13 |
| FRZ-EXE-01, 07 | Governance does not become execution scheduler |
| FRZ-CTR/TYP/PLG/RPL/REG (listed in task) | Plugin evaluator replaceability; registry/catalog proofs where wired |

---

## Unresolved

| Class | Count |
| --- | --- |
| IN-SCOPE BLOCKER | **0** |
| TRACKED FREEZE DEBT | Memory R3-R1 test typing workaround |
| ENVIRONMENT/TEST ISSUE — EVIDENCE REQUIRED | **0** |

---

## Recommended status

```text
GR-12-FINAL = READY FOR AUDIT
GR-12       = READY FOR AUDIT (recommendation; not CLOSED)
GOV-X1      = CURRENT
NEXT        = independent exact-SHA audit → atomic GR-12 closure → next canonical GOV-X1 child
```
