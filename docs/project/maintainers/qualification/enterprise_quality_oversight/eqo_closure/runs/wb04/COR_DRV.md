# EQO-W3-WB04 — COR/DRV disposition paths (wave 9)

**Task ID:** `EQO-W3-WB04-COR`
**COR outcome:** `NOT_APPLICABLE_WITH_EVIDENCE` for product CL-01 in this wave — documentation disposition paths only.
**AUD session E (aggregate):** `BLOCKED` (`runs/wb04/AUD_SESSION_E.md`)
**Evidence pin:** TRAIN_PIN `28c27cb896d02e0185b9fa02047411fa152f6469`
**Ref:** `governance/tickets/T-IGX-046-F.md`

## Source artefacts

| Artefact | Role |
|----------|------|
| `runs/wb04/AUD_SESSION_E.md` | Independent session E verdict (input) |
| `runs/wb04/W3_CLS.md` | Partial CLS pre-close (wave 5) |
| `runs/wb04/COR_DRV.md` | This wave-9 docs routing record |

## AUD packet binding

**Packet:** `AUD_PACKET.md` (`OPEN`, land SHA `8631b1e8459e096b74d1c80247d3e432b44fff30`)

## Finding disposition paths (docs-only @ pin)

| Finding / obligation | AUD verdict | Docs disposition path | Target artefact | Product / EXE route (deferred) |
|----------------------|-------------|----------------------|-----------------|-------------------------------|
| EQOA-A2-047 | The hypothesis is documented at the pin. `PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST.md:239` records `FRZ-CTL-01` as `PASS` (accepted HEAD `318a4d83…`, proof replay 44/44) without mentioning the 15 failures. `CTRL_X_ENTERPRISE_CONTROL_PLANE_RECERTIFICATION.md:385,413,587` records 15 pre-existing Security test failures deferred to QUAL-X (`test_ctrl_x_r2_security_baseline.py`, present at the pin). That file is not in the slice, so there is no execution evidence for the 15 failures at the pin, and the slice's CTRL-X tests cannot refute or confirm them. Closure needs a replay of the baseline file or an owner/operator decision on how `FRZ-CTL-01` states the deferral. | `OWNER_DECISION_RECORD` | eqo_closure/w4/W4-03_OWNER_DECISION_BATCH.md or operator packet | DEFERRED — wave 10 EXE lane (out of wave 9) |
| EQOA-A7-013 | The test that would exercise MCP exposure security did not run (see above). `PDC_TO_ARCH_FREEZE_RESIDUAL_QUALIFICATION.md:84` (EX-D6-04) records the original gap as "MCP exposure security not exercised because fastmcp was absent". The EXE run likely repeated that condition. The hypothesis ("no later source audits it") is neither confirmed nor refuted. Route: replay the P5 batch with the `mcp` extra installed and attach a log that records the per-target exit codes. | `DOC_PATCH` | eqo_closure/runs/*/COR_DRV.md cross-ref for EQOA-A7-013 | DEFERRED — wave 10 EXE lane (out of wave 9) |
| EQOA-A5-015 | No ENT-AUDIT FULL run is in the slice, and the packet cites no run after T-IGX-041-M. The pytest slice cannot decide it. Closure needs a post-041-M ENT-AUDIT FULL run record or a scoped acceptance. | `DOC_PATCH` | freeze checklist / ENT recertification record | DEFERRED — wave 10 EXE lane (out of wave 9) |

## Aggregate rationale (session E)

Session E aggregate **`BLOCKED`** — corrective product work is explicitly out of scope for wave 9; routes above are limited to qualification / governance documentation under `eqo_closure/` and linked owner records.

## DRV trigger (pin / train move)

- After #680 / train pin advances: execute `EQO-W3-WB04-DRV` (`RV-01`); do not open a new EQO campaign.
- Product fixes and pytest replays: wave 10+ lanes per `EQO_WAVE9_MANIFEST.md`.

## Limits and non-claims

- Derived from `AUD_SESSION_E.md` and existing DIS artefacts only; no pytest re-run in this lane.
- No G6 closure, IIAC IN_FORCE, FRZ promotion, or FCS readiness claim.
- No product code, scripts, tests, or register apply in this write set.
