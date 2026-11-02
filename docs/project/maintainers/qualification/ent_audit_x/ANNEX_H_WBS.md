# ANNEX H — Implementation WBS (atomic tasks)

**START_HEAD:** `8a41d32043b89e555d7c18e052cf7022b8c94681`  
**Implementation batches:** 13 (B0–B12)  
**Atomic tasks:** 128  

Format: **ID** | Owner | Path | Input | Change | Deps | Tests | Evidence | Risk | Acceptance

## B0 — Planning publication (8 tasks)

| ID | Path | Change | Tests | Acceptance |
| --- | --- | --- | --- | --- |
| ENT-T-001 | `docs/.../ent_audit_x/` | Publish master + annexes @ HEAD | doc link lint | Independent review READY |
| ENT-T-002 | `ENT_AUDIT_X_SHA_LADDER.md` | Initial ladder rows | manual review | Ladder fields present |
| ENT-T-003 | `ANNEX_C/D/F/G/H/B` | Complete matrices | checklist §4 gate | 13/13 completeness criteria |
| ENT-T-004 | roadmap §5 | **No** stage CLOSED edit | n/a | deferred to B12 |
| ENT-T-005 | `EVOL_X_ENT_AUDIT_X_HANDOFF` | Cross-ref digest pins | diff | reconciled to TES/FINAL |
| ENT-T-006 | operator register | Document OD-ENT-00..08 | n/a | decisions listed |
| ENT-T-007 | `.cursor/plans/` | Archive plan remediation | n/a | pointer to SSOT path |
| ENT-T-008 | completeness audit | Self-audit record | n/a | 13/13 PASS |

## B1 — Contracts/schemas (10)

| ENT-T-101 | `intergrax/platform/enterprise_audit/contracts/` | AuditCheck, Finding, Severity enums | unit schema | pyright clean |
| ENT-T-102 | `schemas/audit_execution_manifest.v1.json` | JSON schema | jsonschema test | validates sample |
| ENT-T-103 | `schemas/enterprise_audit_report.v1.json` | report schema | jsonschema test | validates sample |
| ENT-T-104 | `schemas/coverage_matrix.v1.json` | coverage schema | test | FRZ map slot |
| ENT-T-105 | `schemas/baseline_comparison.v1.json` | drift schema | test | 4 drift classes |
| ENT-T-106 | version field + migration policy doc | contract versioning | test | semver policy |
| ENT-T-107 | tier boundary import test | no agents/apps import | architecture gate | gate PASS |
| ENT-T-108 | public API `__init__.py` exports | stable surface | test | documented |
| ENT-T-109 | CheckDependency graph types | DAG types | unit | cycle detection |
| ENT-T-110 | FrzMapping typed rows | FRZ ID linkage | unit | AUD-03 prep |

## B2 — Runner core (12)

| ENT-T-201 | `runner.py` | EnterpriseAuditRunner | integration | runs 0 checks noop |
| ENT-T-202 | scheduling | topological sort | unit | order stable |
| ENT-T-203 | isolation | per-check try/except | unit | failure isolated |
| ENT-T-204 | timeout | per-check wall clock | unit | timeout→partial |
| ENT-T-205 | cancellation | cooperative cancel | unit | exit 3 on internal |
| ENT-T-206 | parallelism | bounded pool | stress | deterministic order id |
| ENT-T-207 | coverage accounting | expected vs executed | unit | partial flagged |
| ENT-T-208 | result aggregation | AuditResult | unit | severity rollup |
| ENT-T-209 | manifest writer | execution manifest | integration | SHA fields |
| ENT-T-210 | repo identity | remote + branch check | unit | wrong repo→2 |
| ENT-T-211 | git SHA reader | current HEAD | unit | dirty tree flag |
| ENT-T-212 | runner registry injection | DI port | unit | test doubles |

## B3 — Registry + execution safety (14)

| ENT-T-301 | `check_manifest.v1.json` | SSOT allowlist | test | unknown rejected |
| ENT-T-302 | `CheckExecutionSandbox` | subprocess wrapper | adversarial | no shell inject |
| ENT-T-303 | argv allowlist | fixed pytest/uv invocations | test | arbitrary cmd→3 |
| ENT-T-304 | cwd constraint | repo root only | test | path traversal→3 |
| ENT-T-305 | env filter | strip secrets | test | no token leak in logs |
| ENT-T-306 | `PytestGateCheck` adapter | RO profile | integration | PG-003 runs |
| ENT-T-307 | `ScriptGateCheck` adapter | RO scripts | integration | PG-001 runs |
| ENT-T-308 | mutating flag enforcement | refuse mutating in PR FAST | test | policy enforced |
| ENT-T-309 | worktree isolation spec | doc + hook for mutating | test | optional worktree |
| ENT-T-310 | evidence capture dir | `.tmp/proof/...` | test | artifacts written |
| ENT-T-311 | dedupe engine | D1-D5 rules | unit | duplicates merged |
| ENT-T-312 | manifest loader | fail-closed | test | corrupt manifest→3 |
| ENT-T-313 | check authz | registry-only execute | adversarial | unregistered→3 |
| ENT-T-314 | duplicate execution guard | idempotent run ids | unit | same run twice OK |

## B4 — Gate orchestration (16)

| ENT-T-401–416 | Map Annex C area checks ENT-A-*-01..03 + PG-001..020 | AuditCheck rows | L2 | each primary PG GREEN @ SHA |

(Acceptance per row: check id in manifest → adapter executes → finding or clean recorded.)

## B5 — TEN module (10)

| ENT-T-501–510 | TEN-IDENTITY..TEN-FAIL-CLOSED wrappers | pytest delegation | adversarial negatives | 13/13 mapped |

## B6 — VBN module (10)

| ENT-T-601–610 | VBN-* + domain rollup hooks | inventory hash diff | inject stale inventory | 9/9 mapped |

## B7 — Baseline + SHA (10)

| ENT-T-701 | `ENTERPRISE_AUDIT_BASELINE.json` format | baseline schema | validator test | promotion rules |
| ENT-T-702 | drift engine | NEW/RESOLVED/… | unit tests | AUD-07 |
| ENT-T-703 | non-ancestor guard | git merge-base | test | exit 2 |
| ENT-T-704 | tamper detect | digest chain | adversarial | exit 3 |
| ENT-T-705 | qualification_sha vs audit_execution_sha split | manifest fields | test | doc publish safe |
| ENT-T-706 | ladder updater script | optional maintenance | test | updates ladder md |
| ENT-T-707 | EVOL digest reconcile | handoff D10 | integration | pins match |
| ENT-T-708 | baseline promotion CLI subcommand | operator | test | rejects failed FREEZE |
| ENT-T-709 | first baseline proposal | OD-ENT-01 input | doc | not auto promoted |
| ENT-T-710 | requal triggers | material change policy | doc | ties to post-freeze gov |

## B8 — Reporting (8)

| ENT-T-801–808 | JSON builder, MD renderer, FRZ section, limitations section, ownership attribution, CI artifact layout, schema validation, sample golden files |

## B9 — CLI + modes (10)

| ENT-T-901–910 | argparse, FAST/FULL/FREEZE/CHANGED-SINCE, exit 0-3, mode coverage labels, CHANGED-SINCE git diff input, EVOL selection hook, resource budgets, human summary, machine JSON path |

## B10 — CI (8)

| ENT-T-1001 | `.github/workflows/enterprise-audit.yml` | workflow | run | jobs exist |
| ENT-T-1002 | PR FAST required job name | document for OD-ENT-02 | n/a | name stable |
| ENT-T-1003 | nightly FULL | schedule | run | artifact retain |
| ENT-T-1004 | freeze candidate workflow_dispatch FREEZE | manual | run | PASS gate |
| ENT-T-1005 | artifact upload | SHA named | inspect | downloadable |
| ENT-T-1006 | integrate changed paths | pull_request paths | test | CHANGED-SINCE on PR |
| ENT-T-1007 | document branch protection steps | OD-ENT-02 runbook | n/a | exact GitHub UI steps |
| ENT-T-1008 | fail if branch unprotected in FREEZE | test records BLOCKED | test | FRZ-AUD-10 honest |

## B11 — L1/L2/L3 (12)

| ENT-T-1101–1112 | unit/, integration/, l3/, adversarial suites per master plan §17 |

## B12 — Certification + closure (10)

| ENT-T-1201–1210 | Protocol 03 dossier, B6 evidence bundle, corrective batch playbook, FRZ-AUD promotion, OD-ENT-08 record, roadmap §5 row, ARCH-FREEZE handoff doc, remote HEAD verify, operator sign-off, stage CLOSED only after independent PASS |
