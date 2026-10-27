# DIST-SCALE-X — P6 envelope record

| Field | Value |
| --- | --- |
| **Record ID** | `DIST-SCALE-X-ENVELOPE-P6-v1` |
| **Operator authorization** | G-01 + G-03 (WL-7 profile) + G-04/G-05 (physical) |
| **SAC** | `SAC-DIST-SCALE-X-v1` |

## WL-7 soak profile

| Parameter | Value |
| --- | --- |
| Duration | **14400 s** (4 h) — `DIST_SCALE_X_WL7_DURATION_SECONDS` |
| Workers | **2** |
| Tenants | **3** |
| Mix (offered) | 40% WL-1-like, 20% WL-3-like, 15% WL-4-like, 15% WL-6-like @ 5% fault in slice, 10% WL-5-like |
| RSS/handle sample | every **300 s** |
| Queue depth sample | every **60 s** |
| p95 latency sample | every **900 s** |

## Chaos (G-04)

Isolated qual worker SIGKILL; PG `docker pause`; Rabbit `docker stop`/`start`; HTTP 503/slow; event-store deny wrapper; tool slow > deadline.

**Forbidden:** volume delete, shared prod stop, host-wide network, live vendors.
