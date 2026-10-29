# EBH-6 — Evidence Freshness Ledger

| Field | Value |
| --- | --- |
| **CLOSURE_BASELINE_SHA** | `202239f726dd5e3fe23bf1a68cafa61c1eb9fca7` |
| **IMPLEMENTATION_SHA** | **CERTIFICATION_HANDOFF_SHA** |
| **TEST_EXECUTION_SHA** | **CERTIFICATION_HANDOFF_SHA** |
| **CERTIFICATION_HANDOFF_SHA** | @ final enterprise closure wave — see ladder below |
| **PLAN_PUBLICATION_SHA** | `0489054c5f80ee392211fb33a39f2736fa2fdfa0` — [EBH_6_FINAL_BLOCKER_CLOSURE_PLAN.md](EBH_6_FINAL_BLOCKER_CLOSURE_PLAN.md) |

## SHA ladder (CURRENT)

| Label | SHA | Freshness |
| --- | --- | --- |
| **CLOSURE_BASELINE_SHA** (contracts_typing) | `202239f726dd5e3fe23bf1a68cafa61c1eb9fca7` | **CURRENT** implementer evidence anchor |
| **FINAL_CLOSURE_WAVE** IMPLEMENTATION / TEST | `c852e7705f318b5d50a771596d97193f68a00e67` | **CURRENT** @ E09 (213 passed) |
| **PLAN_PUBLICATION_SHA** (corrected closure plan) | `0489054c5f80ee392211fb33a39f2736fa2fdfa0` | **CURRENT** |
| T2 repair handoff | `33b77c7828fff9d2606b2f8cf434b8d9c24a2c86` | Superseded by contracts_typing baseline |

## HISTORICAL — do not cite as CURRENT

| Label | SHA | Note |
| --- | --- | --- |
| B1 ladder HEAD | `83b0a26186b0c11430c916842d7d813554286921` | B2 START_HEAD |
| P03 integrity IMPLEMENTATION | `1f3fb2ae3b8809dd59eb84f0a8d7d8f471aedbd9` | Superseded by B1 ladder |
| L9 wave IMPLEMENTATION / HANDOFF | `0549a12ad53f5128783e38f79fa68ad0d40f00e1` | RC-05 drift — demoted |
| Pre-cert IMPLEMENTATION | `716efd6716a51dc933b85adfe78d6bd13b73bdbf` | Pre-cert anchor |
| L9 plan operator anchor | `d993f49ed3698b1decf399796a6e502b2221caaa` | L9 doc wave |
| P03 remediation IMPLEMENTATION | `475c20289a1ef0420461a1ba748c792a10ea3e99` | Register header historical |
| P03 handoff (EV=0 mask — REJECTED) | `129eee82dff33fd4d815643387a10fa2d56058bf` | Superseded by P03 integrity remediation |
| **P03 integrity IMPLEMENTATION** | `1539c886ddd63f551b62352ad97b182569eb8444` | Superseded by B5 final evidence wave |
| Operator EXPECTED_START_HEAD (P03 integrity) | `f3e39db3f378e5779d5006f54223358237430695` | Superseded |
| STAGE_BASELINE (scope wave) | `8e821c29646a2895b1b3f6e90440cde0b7ea7e05` | Superseded |
| PRE-PUBLICATION local batch | `badb7a086d4d4d4b374535b8698efb06eb8fa6db` | **do not cite as HEAD** |
| EBH-5 independent audit | `33ecf19cc020b112d9e586c87b49e819202f5d6d` | scoped PASS |
| CONFIG-X closed-world PASS | `fac9a43f3b128fde63541d2b56ebd8f4a751e8fb` | preserved |
| CTRL-X independent audit | `318a4d835534ef5c48575f4649551c957dacd8ea` | CLOSED |

## Blockers recorded

- **OD-EBH6-01** OPEN — implementation wave not operator-open
- **OD-EBH6-02** **APPROVED 2026-10-28** — plan execution (corrected closure plan @ `0489054c…`)
- **L9 Protocol 03** OPEN — **T2-AUD** (**T1=0**, **T2=131**, **T3=2**); **OD-EBH6-Q2D2 OPEN**; E10 not executed
- **contracts_typing** @ `202239f72…` — **ACCEPTED** (0× implementer INSUFFICIENT)

## Pytest logs

| Wave | Path |
| --- | --- |
| contracts_typing | `.tmp/session/ebh-6-contracts-typing/pytest.log` @ `202239f72…` |
| T2 repair (historical) | `.tmp/session/ebh-6-t2-repair/pytest.log` |
| **Final closure E09** | `.tmp/session/ebh-6-final-closure/pytest.log` @ **CERTIFICATION_HANDOFF_SHA** |
