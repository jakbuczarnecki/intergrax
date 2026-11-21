# EQO-W3-WB09 — Skills & serving (band C)

**Pin:** `28c27cb896d02e0185b9fa02047411fa152f6469`  
**Band:** C isolated pytest (`eqo-w3`), shared lock `D:/Projekty/_wt/_locks/pytest.lock`.

## Finding scope

| ID | Probe focus |
|----|-------------|
| A2-027 | SKILL-X certified surface changed after acceptance; F-04 scope note obsolete and a PROD-Q … |
| A4-040 | Most Tier-3 application entrypoints start serving with direct uvicorn.run instead of the h… |

## EXE @ TRAIN_PIN

Prereg SHA256: `a9361ab1914706309bc7081f553d335b021854a2a2f808233efc1ba7f0461030` (`DES_PREREG.json`).

**Wave-6 rerun (2026-11-21):** 66 passed in 66.86s (exit 0). Prior wave-1 slice (exit 1) superseded by this log under pytest lock with dev-unit-cert sync.

## Disposition (pre-AUD)

| Finding | DIS (interim) |
|---------|----------------|
| A2-027 | `NOT_VERIFIED` — skills + PROD-Q serving slice green; post-acceptance drift hypothesis open for session E |
| A4-040 | `NOT_VERIFIED` — wiring gate green; uvicorn direct-start inventory not replayed in slice |

Evidence log: `pytest_wb09_exe.log`. AUD session E pending for all rows.