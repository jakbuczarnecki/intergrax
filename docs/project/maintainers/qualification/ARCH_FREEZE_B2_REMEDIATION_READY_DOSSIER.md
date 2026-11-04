# ARCH-FREEZE B2 — remediation-ready backlog (preparation only)

**Prepared @ EXECUTION_SHA:** `18a0f68fc6c8107e6344fdb4641ecec1945dae5e`  
**Not authorized:** implement B2 in wave 1.

## Batch A — inventory gate closure (COMPAT-X)

| Task | Files / domain | Owner | Contract / security | Regression risk | Qualification | Depends on | Gate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| T-B2-CW-GATES (proposed) | `testing_support/change_safety/discovery/*`; qual manifest collect | implementer | CW-05 AST/collect reconciliation | Medium — qual scope | L2: `test_evol_x_closed_world_gates.py` | B1 inventory digest | CW-05..10 PASS @ HEAD |
| T-B2-BND03-RECONCILE | `debug_lab_nexus_loop.py`; checklist FRZ-BND-03 row | implementer | EvidencePersistencePort vs debt text | Low doc; medium if code | L2: layer boundary gate | T-B1-PROD-REACH-SCAN | Evidence reconciled |

## Batch B — debt / doc / VBN (parallel after Batch A partial)

| Task | Files | Owner | Tenant/security | Qualification | Gate |
| --- | --- | --- | --- | --- | --- |
| T-B2-DEBT | debt registers | implementer | governance separation | ENT debt domain L2 | FRZ-DEBT rollup |
| T-B3-DOC | manifests under `docs/project` | implementer | doc canon | replay_x mandatory docs | FRZ-DOC |
| T-B2-VBN | `scripts/enterprise_audit/vbn*` | implementer | vendor neutrality | ENT VBN 9 checks | FRZ-RPL |

## Batch C — external formal (no agent self-cert)

| Task | Blocker | Qualification |
| --- | --- | --- |
| T-B5-EVOL-B6 / T-B6-ENT-B6 | Independent auditor | L3 EVOL/ENT TES replay |
| T-B6-OD-ENT-* | Operator / GitHub admin | OD packets |

## Safe parallelism

- **Parallel:** T-B3-DOC with T-B2-VBN after T-B2-BND03-RECONCILE starts (no shared files).
- **Serial:** CW gate closure before certifying ENT FULL/FREEZE replays.

## Acceptance before B3 qualification wave

1. `verify_arch_freeze_plan_traceability.py` exit 0 (including Path B PRD split).
2. B1 inventory reproduced @ pinned EXECUTION_SHA with identical `inventory_digest`.
3. CW gates PASS or evidence-backed waivers recorded in gap register (operator only).
