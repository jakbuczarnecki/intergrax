# DIST-SCALE-X — Dimension M Evidence (P6)

| Field | Value |
| --- | --- |
| **Envelope** | `DIST-SCALE-X-ENVELOPE-P6-v1` |

## WL-7 soak

| Item | Value |
| --- | --- |
| Test | `test_p6_wl07_mixed_soak_stability` |
| Original P6 lock | 14400 s — **NOT conducted** (see amendment) |
| **1 h substitute** | 3600 s — both `DIST_SCALE_X_WL7_DURATION_SECONDS` and `DIST_SCALE_X_WL7_MIN_ENVELOPE_SECONDS` |

**Exit 11 (amendment path):** `soak_complete_envelope=1` @ **3600 s** — [DIST_SCALE_X_WL7_ENVELOPE_AMENDMENT_1H_RECORD.md](DIST_SCALE_X_WL7_ENVELOPE_AMENDMENT_1H_RECORD.md).

## Harness @ `9ac74ca…`

| run_id | duration_s | soak_complete_envelope | errors |
| --- | --- | --- | --- |
| `wl07-soak-41ad330f8ef2` | 120 | 0 | 0 |

## Physical 1 h substitute @ `9a062322826de8444d430eed1556d7463a2cd2bf`

| run_id | duration_s | soak_complete_envelope | errors |
| --- | --- | --- | --- |
| `wl07-soak-c8e86ebef60e` | 3600 | **1** | 0 |

Manifest: `.tmp/proof/dist-scale-x/wl07-soak-c8e86ebef60e/`. **4 h NOT PROVEN.**
