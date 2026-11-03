# ANNEX F — Exact AS-IS gap register

**START_HEAD / PLAN_BASELINE_SHA:** `8a41d32043b89e555d7c18e052cf7022b8c94681`  
**Corrective reconciliation @ ENT-AUDIT-X wave:** execution SSOT [ENT_AUDIT_X_POST_IMPLEMENTATION_FINAL_CLOSURE_MASTER_PLAN.md](./ENT_AUDIT_X_POST_IMPLEMENTATION_FINAL_CLOSURE_MASTER_PLAN.md) — **GAP-RC (33)** tracked in Matrix D; formal **GAP-B (6)** remain operator/auditor gates. Mechanical plan validator: `scripts/enterprise_audit/validate_post_implementation_master_plan.py`.

## Separate counts (not merged)

| Bucket | Count |
| --- | ---: |
| GAP-C missing components | 14 |
| GAP-P partial mechanisms | 9 |
| Mandatory check definitions (Annex C) | 85 |
| GAP-COV coverage deficiencies (no ENT adapter @ PLAN_BASELINE) | 85 |
| GAP-D documentation gaps | 8 |
| GAP-B formal blockers | 6 |
| GAP-O external/operator dependencies | 9 |

## GAP-C — missing components

| GAP ID | Absence evidence | Remediation | Owner | WBS | Closure test |
| --- | --- | --- | --- | --- | --- |
| GAP-C-01 | No `intergrax/platform/enterprise_audit/` | Create Tier-0 package | ENT | ENT-T-0101 | import boundary gate |
| GAP-C-02 | No `EnterpriseAuditRunner` | Runner module | ENT | ENT-T-0201 | `test_ent_audit_runner_exists` |
| GAP-C-03 | No check manifest SSOT | `enterprise_audit_check_manifest.v1.json` | ENT | ENT-T-0301 | manifest schema test |
| GAP-C-04 | No subprocess sandbox | `CheckExecutionSandbox` allowlist | ENT | ENT-T-0302 | injection adversarial |
| GAP-C-05 | No baseline store | `ENTERPRISE_AUDIT_BASELINE.json` | ENT/Op | ENT-T-0701 | baseline validator |
| GAP-C-06 | No drift engine | Drift classifier | ENT | ENT-T-0702 | drift semantics tests |
| GAP-C-07 | No report builder | JSON/MD + schemas | ENT | ENT-T-0801 | schema tests |
| GAP-C-08 | No CLI | `scripts/enterprise_audit/run_enterprise_audit.py` | ENT | ENT-T-0901 | exit code tests |
| GAP-C-09 | No mode policies | FAST/FULL/FREEZE/CHANGED-SINCE | ENT | ENT-T-0902 | mode matrix test |
| GAP-C-10 | No CI workflow | `enterprise-audit.yml` | ENT | ENT-T-1001 | workflow run |
| GAP-C-11 | No qual tree | `tests/qualification/ent_audit_x/` | ENT | ENT-T-1101 | L1/L2/L3 |
| GAP-C-12 | No Protocol 03 dossier | `ENT_AUDIT_X_PROTOCOL_03_*` | ENT | ENT-T-1201 | auditor checklist |
| GAP-C-13 | No composite evidence bundle | `.tmp/proof/ent-audit-x/` layout spec | ENT | ENT-T-1202 | bundle manifest |
| GAP-C-14 | No operator closure record | `OD-ENT-08` template | Operator | ENT-T-1203 | signed record |

## GAP-P — partial

| GAP ID | Partial state | WBS |
| --- | --- | --- |
| GAP-P-01 | CI `--profile enterprise` ≠ platform audit | ENT-T-1002 |
| GAP-P-02 | 411 gate files without manifest index | ENT-T-0303 |
| GAP-P-03 | Post-freeze guard matrix stale header SHA | ENT-T-1102 |
| GAP-P-04 | EVOL W7-06 handoff pin TBD | EXT-EVOL-06 |
| GAP-P-05 | `development` branch not protected (gh API 404) | OD-ENT-02 |
| GAP-P-06 | EBH-5 reachability graph skipped @ historical baseline | ENT-T-0608 |
| GAP-P-07 | Overlapping ideal/ebh gates | ENT-T-0304 dedupe rules |
| GAP-P-08 | Historical PLATFORM_WIDE audit | reference-only flag |
| GAP-P-09 | EVOL selection not bound to CHANGED-SINCE | ENT-T-0903 |

## GAP-B — formal blockers

| ID | Resolution path |
| --- | --- |
| BLK-EVOL-B6 | Human Protocol 03 @ EVOL TES/FINAL; publish B6 SHA |
| BLK-EVOL-FRZ | W7-03 promotes FRZ-EVOL after B6 |
| BLK-EVOL-HANDOFF | W7-06 updates handoff closure SHA |
| BLK-CI-ENFORCE | GitHub branch protection + required checks |
| BLK-ROADMAP-ENT-DEP | Roadmap requires EVOL CLOSED for ENT stage closure — operator decision **OD-ENT-00** (not waived in this plan) |
| BLK-ENT-B6 | Independent ENT Protocol 03 before FRZ-AUD PASS |

## GAP-COV

Each of **85** Annex C checks: **GAP-COV-&lt;CheckID&gt;** = no `AuditCheck` adapter @ HEAD; closes when corresponding WBS batch lands and L2 test passes.
