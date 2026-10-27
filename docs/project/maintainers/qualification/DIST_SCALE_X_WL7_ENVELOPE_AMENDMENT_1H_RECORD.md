# DIST-SCALE-X — WL-7 envelope amendment (1 h substitute)

| Field | Value |
| --- | --- |
| **Record ID** | `DIST-SCALE-X-WL7-ENVELOPE-AMENDMENT-1H-2026-10-27` |
| **Date** | 2026-10-27 |
| **Operator intent** | **Full 14400 s (4 h) WL-7 physical soak NOT conducted** in this qualification cycle; authorize **3600 s (1 h)** substitute soak for dimension **M** with matching min-envelope env pins |
| **SAC** | `SAC-DIST-SCALE-X-v1` — **operator envelope amendment for WL-7 duration only** (not a global SAC rewrite); original 4 h lock in [DIST_SCALE_X_P6_ENVELOPE_RECORD.md](DIST_SCALE_X_P6_ENVELOPE_RECORD.md) preserved in history |
| **Supersedes (execution goal)** | Sibling / build-session **14400 s** runs — cancelled `OPERATOR_CANCEL_4H` |

## Envelope pins (1 h substitute)

| Variable | Value |
| --- | --- |
| `DIST_SCALE_X_WL7_DURATION_SECONDS` | **3600** |
| `DIST_SCALE_X_WL7_MIN_ENVELOPE_SECONDS` | **3600** |

Per `wl_07_soak_mixed_load.py`: `soak_complete_envelope=1` only when `duration >= min_envelope` (both **3600** for this path).

## Honest scope

| Claim | Allowed |
| --- | --- |
| WL-7 **1 h** physical soak **PROVEN** (if PASS, manifest `soak_complete_envelope=1`, `errors=0`) | **Yes** — under this amendment |
| Original P6 **14400 s** lock satisfied without further operator action | **No** — **4 h NOT PROVEN** |
| Parent **CLOSED**, **G-07**, global **FRZ PASS** | **Forbidden** without **G-07** after **G-06** |

## L9 matrix disposition (this cycle)

| Row | Disposition |
| --- | --- |
| **11** (soak M) | **PARTIAL / VALIDATION** — **1 h PROVEN** (if pass) with footnote **4 h NOT conducted**; not the pre-amendment **14400 s** bar |
| **16** (blockers) | Per matrix rules after row **11** refresh — **not** full **16/16** closure on original 4 h envelope unless operator later amends or completes 4 h |

## Session evidence

| Session | Role |
| --- | --- |
| `.tmp/session/dist-scale-x-wl7-1h-substitute/` | **3600 s** substitute run (`run.log`, `launch.env`) |
| `.tmp/session/dist-scale-x-wl7-build-2026-10-27/` | **4 h** cancelled — `poll.log` entry `OPERATOR_CANCEL_4H` |
| `.tmp/proof/dist-scale-x/wl07-soak-c8e86ebef60e/` | **PASS** @ `EXECUTION_SHA` `9a062322826de8444d430eed1556d7463a2cd2bf`; `soak_complete_envelope=1`; `min_envelope_seconds=3600`; `error_count=0` |

## Related

- [DIST_SCALE_X_P6_ENVELOPE_RECORD.md](DIST_SCALE_X_P6_ENVELOPE_RECORD.md) — amendment section
- [DIST_SCALE_X_WL7_PHYSICAL_DEFERRAL_RECORD.md](DIST_SCALE_X_WL7_PHYSICAL_DEFERRAL_RECORD.md)
- [DIST_SCALE_X_INTERIM_SCOPED_QUALIFICATION_RECORD.md](DIST_SCALE_X_INTERIM_SCOPED_QUALIFICATION_RECORD.md)
