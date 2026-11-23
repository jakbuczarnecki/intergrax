# EQO-W3-WB10 — COR/DRV disposition paths (wave 9)

**Task ID:** `EQO-W3-WB10-COR`
**COR outcome:** `NOT_APPLICABLE_WITH_EVIDENCE` for product CL-01 in this wave — documentation disposition paths only.
**AUD session E (aggregate):** `FAIL` (`runs/wb10/AUD_SESSION_E.md`)
**Evidence pin:** TRAIN_PIN `28c27cb896d02e0185b9fa02047411fa152f6469`
**Ref:** `governance/tickets/T-IGX-046-F.md`

## Source artefacts

| Artefact | Role |
|----------|------|
| `runs/wb10/AUD_SESSION_E.md` | Independent session E verdict (input) |
| `runs/wb10/W3_CLS.md` | Partial CLS pre-close (wave 5) |
| `runs/wb10/COR_DRV.md` | This wave-9 docs routing record |

## AUD packet binding

**Packet:** `AUD_PACKET.md` (`OPEN`, land SHA `8631b1e8459e096b74d1c80247d3e432b44fff30`)

## Finding disposition paths (docs-only @ pin)

| Finding / obligation | AUD verdict | Docs disposition path | Target artefact | Product / EXE route (deferred) |
|----------------------|-------------|----------------------|-----------------|-------------------------------|
| EQOA-A2-019 | The certified TRACE-X-P6 restart/resume surface does not hold at the pin. The P6 closed-world gates q02, q08 and q11 fail on an unclassified path, so the certified claim of zero unclear restart/resume paths is contradicted. No TRACE-X requalification record covers the change (the slice does not produce one). Route: owner classifies the path and runs the TRACE-X requalification on a new pin. | `DOC_PATCH` | eqo_closure/runs/*/COR_DRV.md cross-ref for EQOA-A2-019 | NONE — docs path sufficient at pin |
| EQOA-A2-030 | The acceptance of ROADMAP-REPLAY-X depends on gates that are red at the pin: product delta since DUP-X is not empty, closure counters are not zero (3 != 0) and the B6 R2 reconciliation is not zero. The accepted state ("CLOSED — INDEPENDENTLY ACCEPTED (SCOPED)", asserted by `test_replay_x_operator_closure_recorded`, which passes) therefore does not describe the pin. The second half of the hypothesis (acceptance rests on auditor log inspection of artefacts absent from the repo) is not decided here: DUP-X and the EBH-7 GHA run are not in the slice. | `DOC_PATCH` | eqo_closure/ or owner qualification record | DEFERRED — product CL-01 / dedicated ticket (out of wave 9) |
| EQOA-A5-012 | Hypothesis supported by execution at the pin. The dossier says the STATE-X and TRACE-X closed-world gates were not rerun after the 2026-10-30 replay while 274 product files changed; this session did not re-derive the 274 count, but the prod-delta gate itself lists well over a hundred `intergrax/` paths outside its scoped list. Run now, the TRACE-X closed-world gates fail (q02, q08, q11) and the STATE-X P0 baseline passes. The drift is therefore real and has already broken a gate. The "new MCP ingress" and "new durable stores" part is consistent with the listed paths (`intergrax/applications/_shared/mcp_*`, `fastapi_mcp.py`, durable store modules) but the log does not show whether each is classified. | `OWNER_DECISION_RECORD` | eqo_closure/w4/W4-03_OWNER_DECISION_BATCH.md or operator packet | DEFERRED — product CL-01 / dedicated ticket (out of wave 9) |

## Aggregate rationale (session E)

Session E aggregate **`FAIL`** — corrective product work is explicitly out of scope for wave 9; routes above are limited to qualification / governance documentation under `eqo_closure/` and linked owner records.

## DRV trigger (pin / train move)

- After #680 / train pin advances: execute `EQO-W3-WB10-DRV` (`RV-01`); do not open a new EQO campaign.
- Product fixes and pytest replays: wave 10+ lanes per `EQO_WAVE9_MANIFEST.md`.



## DRV disposition (wave 14, docs-only)

**Task ID:** `EQO-W3-WB10-DRV`
**Outcome:** `DISPOSITIONED_NOT_CLOSED` — open `DOC_PATCH` / `OWNER_DECISION_RECORD` routes satisfied with documented deferral only; **no** product DRV replay @ pin.
**Lane:** `eqo/wave-14/drv`

| Route class | Wave-14 status |
|-------------|----------------|
| `DOC_PATCH` | Open — train ticket or post-FCS CL-01; cross-ref `COR_DRV` finding table |
| `OWNER_DECISION_RECORD` | `WAIT_FOR_OPERATOR` — `w4/W4-03_OWNER_DECISION_BATCH.md` |
| Product / pytest DRV | `DEFERRED` — `RV-01` on pin advance; not executed in wave 14 |

Session E verdict **unchanged**. No pytest in this lane.

## Limits and non-claims

- Derived from `AUD_SESSION_E.md` and existing DIS artefacts only; no pytest re-run in this lane.
- No G6 closure, IIAC IN_FORCE, FRZ promotion, or FCS readiness claim.
- No product code, scripts, tests, or register apply in this write set.
