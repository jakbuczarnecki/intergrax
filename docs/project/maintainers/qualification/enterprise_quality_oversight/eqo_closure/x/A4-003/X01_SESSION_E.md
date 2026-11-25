# EQO-X-01 — Session E disposition (tier boundary / WB05)

**Task:** `EQO-X-01` · **Source SHA:** `8bdb0038d7a8fe506b5490a0ec2acad227e6b938`

## Binding reads

| Artefact | Finding |
|----------|---------|
| `runs/wb05/AUD_SESSION_E.md` | Aggregate **BLOCKED** @ TRAIN_PIN evidence slice |
| `runs/wb05/COR_DRV.md` | Session E aggregate BLOCKED; no substitute PASS |
| `x/A4-003/A4_GATE_COUNTER_READ.md` | `MERGED_TO_TRAIN` fact; replay open |

## Session E verdict

**`EXACT_BLOCKER`** — file scan and train merge facts do **not** substitute for **W2-02 named gate execution** PASS @ applicable pin. No pytest invented this session.

**Ref:** `governance/tickets/T-IGX-046-F.md`