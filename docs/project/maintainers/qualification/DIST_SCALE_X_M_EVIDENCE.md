# DIST-SCALE-X — Dimension M Evidence (P6)

| Field | Value |
| --- | --- |
| **Envelope** | `DIST-SCALE-X-ENVELOPE-P6-v1` |

## WL-7 soak

| Item | Value |
| --- | --- |
| Test | `test_p6_wl07_mixed_soak_stability` |
| Full envelope duration | 14400 s (`DIST_SCALE_X_WL7_DURATION_SECONDS`) |
| Harness default | 120 s + skip until full duration configured |

**Exit 11:** requires `soak_complete_envelope=1` in manifest metrics.

## Session result @ EXECUTION_SHA `9ac74ca3849013e729dbc8c893f64427ef588f94`

| run_id | duration_s | soak_complete_envelope | errors |
| --- | --- | --- | --- |
| `wl07-soak-41ad330f8ef2` | 120 | 0 | 0 |

**Blocker:** operator must run `DIST_SCALE_X_WL7_DURATION_SECONDS=14400` for exit 11 closure. Harness correctness **PASS**; envelope duration **NOT MET**.
