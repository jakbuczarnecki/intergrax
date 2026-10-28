# EBH-5 — VENDOR_BACKEND_DEPENDENCY_INVENTORY

| Field | Value |
| --- | --- |
| **Artifact** | `VENDOR_BACKEND_DEPENDENCY_INVENTORY` |
| **Producing stage** | EBH-5 |
| **Evidence SHA (SSOT code)** | `5c9b66db632976ada039d406474530d7c6cb5f86` (baseline; refresh at closure commit) |
| **Machine SSOT** | [`tests/qualification/ebh_5/_vendor_backend_inventory.py`](../../../tests/qualification/ebh_5/_vendor_backend_inventory.py) |
| **Status** | **READY FOR INDEPENDENT AUDIT** — not parent CLOSED |

## Acceptance

| Criterion | Value |
| --- | --- |
| `concern_records` | **69** |
| `unclassified` | **0** |
| CONFIG-X reconciliation | **54/54** mapped (`fac9a43f…` historical PASS preserved) |
| Roadmap §3.0.8 domains 1–31 | Covered via `RoadmapVendorDomain` + `DISCOVERED` |

## Structure

- **54** records derived from [`CONFIG_X_CONCERN_INVENTORY`](../../../tests/qualification/config_x/_config_x_concern_inventory.py) (integration categories + platform activation concerns).
- **15** additional runtime/platform concerns ([`_runtime_platform_concerns.py`](../../../tests/qualification/ebh_5/_runtime_platform_concerns.py)): runtime SQLite persistence ports, single-node profile v1, distributed KV, PROD-Q F02-14 debt row, lab adaptive stores (non-production-effective).

## Verification

```text
uv run pytest tests/qualification/ebh_5/test_ebh_5_l2_integrity_gates.py -q
```

Independent audit MUST re-run at exact closure SHA and reconcile row-level fields against code reachability.
