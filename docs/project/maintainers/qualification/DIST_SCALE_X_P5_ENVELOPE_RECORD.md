# DIST-SCALE-X — P5 envelope record

| Field | Value |
| --- | --- |
| **Record ID** | `DIST-SCALE-X-ENVELOPE-P5-v1` |
| **Operator authorization** | `OPERATOR APPROVED — DIST-SCALE-X P5–L9 INTEGRATED AGENT EXECUTION` (G-01) + P5 parameter grant (G-02) |
| **SAC** | `SAC-DIST-SCALE-X-v1` (OD-03) |
| **Lock mode** | `ENVELOPE_ONLY` / `NOT_LOCKED` |

## Infrastructure

| Service | Version |
| --- | --- |
| PostgreSQL | `postgres:16.6` (host 5434→5432) |
| RabbitMQ | `rabbitmq:3.13-management` |
| HTTP | local loopback (WL-4 reuse) |
| Host | LKW multiprocess workers |

## Scope

Dimensions **I**, **J**, **L** only. Workers **2** baseline, **4** where WL-3-class scenario requires. Tenants **2–3**.

## Manifest binding

`qualification_phase: P5`; F02 env/topology equality; `catalog_integrity_sha256` @ EXECUTION_SHA.
