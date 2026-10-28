# EBH-6 — Evidence Freshness Ledger

| Field | Value |
| --- | --- |
| **B2 EXPECTED_START_HEAD** | `83b0a26186b0c11430c916842d7d813554286921` |
| **IMPLEMENTATION_SHA** | *(set @ B2 qual commit)* |
| **TEST_EXECUTION_SHA** | *(set @ B2 qual commit)* |
| **CERTIFICATION_HANDOFF_SHA** | `3f0a25439270b173c63989d2fd948c82a8cff6af` — **pre-B1** (unchanged until B5 publication) |
| **PLAN_PUBLICATION_SHA** | `0489054c5f80ee392211fb33a39f2736fa2fdfa0` — [EBH_6_FINAL_BLOCKER_CLOSURE_PLAN.md](EBH_6_FINAL_BLOCKER_CLOSURE_PLAN.md) |

## SHA ladder (CURRENT @ B2)

| Label | SHA | Freshness |
| --- | --- | --- |
| B2 **EXPECTED_START_HEAD** | `83b0a26186b0c11430c916842d7d813554286921` | **CURRENT** |
| B2 **IMPLEMENTATION_SHA** / **TEST_EXECUTION_SHA** | *(set @ B2 qual commit)* | **CURRENT** after B2 ladder commit |
| **PLAN_PUBLICATION_SHA** (corrected closure plan) | `0489054c5f80ee392211fb33a39f2736fa2fdfa0` | **CURRENT** |
| B1 **IMPLEMENTATION_SHA** (historical) | `446dfa7263b6571a11bdd2a973e66e57892eacca` | Superseded by B2 wave |
| **CERTIFICATION_HANDOFF_SHA** | `3f0a25439270b173c63989d2fd948c82a8cff6af` | **pre-B1 handoff** — update @ B5 only |

## HISTORICAL — do not cite as CURRENT

| Label | SHA | Note |
| --- | --- | --- |
| B1 ladder HEAD | `83b0a26186b0c11430c916842d7d813554286921` | B2 START_HEAD |
| P03 integrity IMPLEMENTATION | `1f3fb2ae3b8809dd59eb84f0a8d7d8f471aedbd9` | Superseded by B1 ladder |
| L9 wave IMPLEMENTATION / HANDOFF | `0549a12ad53f5128783e38f79fa68ad0d40f00e1` | RC-05 drift — demoted |
| Pre-cert IMPLEMENTATION | `716efd6716a51dc933b85adfe78d6bd13b73bdbf` | Pre-cert anchor |
| L9 plan operator anchor | `d993f49ed3698b1decf399796a6e502b2221caaa` | L9 doc wave |
| P03 remediation IMPLEMENTATION | `475c20289a1ef0420461a1ba748c792a10ea3e99` | Register header historical |
| P03 final disposition | `f87a6dfc988bb720b6e7475e3cffe279d9290198` | HISTORICAL |
| Operator EXPECTED_START_HEAD (P03 integrity) | `f3e39db3f378e5779d5006f54223358237430695` | Superseded |
| STAGE_BASELINE (scope wave) | `8e821c29646a2895b1b3f6e90440cde0b7ea7e05` | Superseded |
| PRE-PUBLICATION local batch | `badb7a086d4d4d4b374535b8698efb06eb8fa6db` | **do not cite as HEAD** |
| EBH-5 independent audit | `33ecf19cc020b112d9e586c87b49e819202f5d6d` | scoped PASS |
| CONFIG-X closed-world PASS | `fac9a43f3b128fde63541d2b56ebd8f4a751e8fb` | preserved |
| CTRL-X independent audit | `318a4d835534ef5c48575f4649551c957dacd8ea` | CLOSED |

## Blockers recorded

- **OD-EBH6-01** OPEN — implementation wave not operator-open
- **OD-EBH6-02** **APPROVED 2026-10-28** — plan execution (corrected closure plan @ `0489054c…`)
- **L9 Protocol 03** OPEN — **no B6** until B5 census **0** `EVIDENCE_REQUIRED` groups + B3 §3.B gate
- **109** A–J gap groups @ B2 IMPLEMENTATION_SHA (133 applicable pairs — normative crosswalk)

## Pytest log (B2)

Consolidated L1+L2 @ **TEST_EXECUTION_SHA**: `.tmp/session/ebh-6-b2/pytest.log`
