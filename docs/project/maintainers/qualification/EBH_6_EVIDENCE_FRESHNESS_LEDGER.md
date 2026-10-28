# EBH-6 — Evidence Freshness Ledger

| Field | Value |
| --- | --- |
| **STAGE_BASELINE / EXPECTED_START_HEAD** | `4736e4e4a00aff44afea9a28f91d188631ae43f1` |
| **IMPLEMENTATION_SHA** | `475c20289a1ef0420461a1ba748c792a10ea3e99` |
| **TEST_EXECUTION_SHA** | `475c20289a1ef0420461a1ba748c792a10ea3e99` |
| **CERTIFICATION_HANDOFF_SHA** | `475c20289a1ef0420461a1ba748c792a10ea3e99` |
| **PUBLICATION_SHA (P03 remediation)** | `54a77fb83a8e12db0b277f19f64f91684d8f3240` |

## SHA ladder

| Label | SHA | Freshness |
| --- | --- | --- |
| L9 IMPLEMENTATION / TEST_EXECUTION / HANDOFF | `475c20289a1ef0420461a1ba748c792a10ea3e99` | **CURRENT** |
| L9 wave (pre-P03 reject) | `bc25e27b14b7172ef79951765e76d19578d9beb3` | **HISTORICAL** |
| L9 publication doc wave | `ce148559539583f4569d7a43c245e6c830e60a06` | **HISTORICAL** |
| Operator **EXPECTED_START_HEAD** | `4736e4e4a00aff44afea9a28f91d188631ae43f1` | **CURRENT** (execution start) |
| Plan doc anchor (historical note) | `d993f49ed3698b1decf399796a6e502b2221caaa` | **HISTORICAL** — superseded by wave-2 HEAD |
| Pre-cert IMPLEMENTATION | `716efd6716a51dc933b85adfe78d6bd13b73bdbf` | **HISTORICAL** |
| EBH-6 publication tree (P0–P6) | `50d3b7785c7e13dbc6708143e7484a2eeea5eaca` | **HISTORICAL** |
| PRE-PUBLICATION local batch | `badb7a086d4d4d4b374535b8698efb06eb8fa6db` | **HISTORICAL — do not cite as HEAD** |
| EBH-5 independent audit | `33ecf19cc020b112d9e586c87b49e819202f5d6d` | **HISTORICAL — scoped PASS** |
| CONFIG-X closed-world PASS | `fac9a43f3b128fde63541d2b56ebd8f4a751e8fb` | **HISTORICAL — preserved** |
| CTRL-X independent audit | `318a4d835534ef5c48575f4649551c957dacd8ea` | **HISTORICAL — CLOSED** |

## Blockers recorded

- **OD-EBH6-01** OPEN — implementation wave not operator-open
- **OD-EBH6-02** **APPROVED 2026-10-28** — L9 evidence execution (not independent code acceptance)
- **L9 Protocol 03** **READY FOR RE-AUDIT** @ `475c20289a1ef0420461a1ba748c792a10ea3e99` — prior B-L9-5 reject remediated (qual/docs)

## Pytest log (L4)

Consolidated bundle @ **TEST_EXECUTION_SHA**: `.tmp/session/ebh-6-p03-remediation/pytest.log`
