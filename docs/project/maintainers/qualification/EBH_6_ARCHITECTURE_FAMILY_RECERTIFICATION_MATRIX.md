# EBH-6 — Architecture-Family Recertification Matrix

| Field | Value |
| --- | --- |
| **IMPLEMENTATION_SHA** | `badb7a086d4d4d4b374535b8698efb06eb8fa6db` |
| **Mechanical SSOT** | [`tests/qualification/ebh_6/_family_catalog.py`](../../../tests/qualification/ebh_6/_family_catalog.py) (`EBH6_FAMILY_CATALOG`) |
| **Independent disposition** | **PENDING (L9)** for most families — not global FRZ PASS |

Per-family columns live in catalog dataclass: Authority, Scope paths, Existing evidence SHA, Freshness, Missing proof, Gate/test IDs, Remediation batch, Acceptance rule, Independent disposition.

## Applicable families (@ HEAD)

| family_id | ecap_theme | frz_family | independent_disposition |
| --- | --- | --- | --- |
| BND | Hard layer boundaries | FRZ-BND-* | REPLAY PASS |
| OWN | Exactly-one ownership | FRZ-OWN-* | PENDING |
| CTR | Canonical contracts | FRZ-CTR-* | PENDING |
| TYP | Strong typing | FRZ-TYP-* | PENDING |
| PLG | Pluginability | FRZ-PLG-* | REPLAY PASS |
| RPL | Replaceability | FRZ-RPL-* | REPLAY PASS |
| EXE | Execution | FRZ-EXE-* | PENDING |
| GOV | Governance | FRZ-GOV-* | PENDING |
| STA | State | FRZ-STA-* | PENDING |
| REC | Recovery | FRZ-REC-* | PENDING |
| CMP | Compatibility | FRZ-CMP-* | N/A — WITH EVIDENCE |
| CFG | Configuration | FRZ-CFG-* | REPLAY PASS |
| SEC | Security | FRZ-SEC-* | PENDING |
| REL | Reliability | FRZ-REL-* | PENDING |
| CTL | Control planes | FRZ-CTL-* | REPLAY PASS |
| OBS | Observability | FRZ-OBS-* | PENDING |
| TRC | Traceability | FRZ-TRC-* | PENDING |
| PRD | Production qual (arch) | FRZ-PRD-* | N/A — WITH EVIDENCE |
| REG | Regression infra | FRZ-REG-* | PENDING |
| HRN | Harness | FRZ-HRN-* | PENDING |
| TEN | Tenant | FRZ-TEN-* | PENDING |
| DOC | Documentation | FRZ-DOC-* | PENDING |
| VBN | Vendor neutrality rollup | (VBN) | PENDING |

**Row count:** 23 applicable families (closed-world catalog).

## Guard / gate replay

Selected pytest bundles per family — see `gate_test_ids` in catalog. Layer boundary gate: [`test_hardening_3_layer_boundary_gate.py`](../../../tests/unit/runtime/architecture/test_hardening_3_layer_boundary_gate.py). EBH-5 RPL-G + inventory integrity replay @ HEAD.

## Explicit non-claims

- Matrix disposition **≠** global [`PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST.md`](PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST.md) PASS rewrite.
- **51** EBH-5 matrix rows **NOT YET QUALIFIED** remain deferred (EBH-7).
