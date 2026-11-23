# EQO-W3-WB01 — COR/DRV disposition paths (wave 9)

**Task ID:** `EQO-W3-WB01-COR`
**COR outcome:** `NOT_APPLICABLE_WITH_EVIDENCE` for product CL-01 in this wave — documentation disposition paths only.
**AUD session E (aggregate):** `FAIL` (`runs/wb01/AUD_SESSION_E.md`)
**Evidence pin:** TRAIN_PIN `28c27cb896d02e0185b9fa02047411fa152f6469`
**Ref:** `governance/tickets/T-IGX-046-F.md`

## Source artefacts

| Artefact | Role |
|----------|------|
| `runs/wb01/AUD_SESSION_E.md` | Independent session E verdict (input) |
| `runs/wb01/W3_CLS.md` | Partial CLS pre-close (wave 5) |
| `runs/wb01/COR_DRV.md` | This wave-9 docs routing record |

## AUD packet binding

**Packet:** `AUD_PACKET.md` (`OPEN`, land SHA `8631b1e8459e096b74d1c80247d3e432b44fff30`)

## Finding disposition paths (docs-only @ pin)

| Finding / obligation | AUD verdict | Docs disposition path | Target artefact | Product / EXE route (deferred) |
|----------------------|-------------|----------------------|-----------------|-------------------------------|
| EQOA-A1-021 | Hypothesis confirmed. `PLATFORM_ENTERPRISE_COMPLETION_ROADMAP.md:1939` still says "Independently audited establishment of the enterprise-freeze control system" and cites only commit `324ad07d`. A search of `docs/` and `governance/` at the pin finds that SHA only in the roadmap, the FRZ checklist (EBH-2F-R1 row) and `ROADMAP_REPLAY_X_HISTORICAL_STAGE_MATRIX.md:28`, which rates it `RULE-ACCEPTED-CHILD`. No audit report exists. The disposition names no scope, owner or reopen condition. Route: owner relabels L1939 or cites the audit record (DOC_PATCH). | `DOC_PATCH` | eqo_closure/ or owner qualification record | DEFERRED — product CL-01 / dedicated ticket (out of wave 9) |
| EQOA-A4-044 | Defect reproduced by reading source. `intergrax/runtime/execution/nexus_host_execution.py:57-60` builds `AgentRouter(nexus_loop.registry, event_bus=nexus_loop.event_bus)` with no `production_mode`, so it takes the default `False` (`agent_router.py:49`). `NexusLoop` passes `production_mode=production_mode` to its own router (`nexus_loop.py:361-364`). Host-task routing therefore ignores `production_mode`. T-IGX-043-A / #680 is not on the pin, so the fix is not present. The slice has no routing test. Route: land #680 (T-IGX-043-A), then replay. | `DOC_PATCH` | eqo_closure/runs/*/COR_DRV.md cross-ref for EQOA-A4-044 | DEFERRED — product CL-01 / dedicated ticket (out of wave 9) |
| EQOA-A4-015 | Relaxed construction exists at the pin: `agents/organization_worker/organization_worker_agent.py:114` hard-codes `production_mode=False`, `agents/boundary_demo/boundary_demo_agent.py:87` uses the lab config, and `agents/external_contractor_adapter/external_contractor_adapter_agent.py:113` uses `harness_production_mode()`. Whether any production host reaches these paths is the open question, and no test in the slice answers it. It also depends on #680. | `DOC_PATCH` | eqo_closure/runs/*/COR_DRV.md cross-ref for EQOA-A4-015 | DEFERRED — product CL-01 / dedicated ticket (out of wave 9) |
| EQOA-A5-003 | No gate-weakening review for the GOV-X1 rewrite was found under `docs/`, `governance/` or `tests/` at the pin. The hits are the findings register and unrelated rows. Proving a negative needs the owner (PDC-X D6 / ARCH-FREEZE) to produce the review or record the limitation with scope and reopen condition. LOW severity. | `DOC_PATCH` | eqo_closure/runs/*/COR_DRV.md cross-ref for EQOA-A5-003 | NONE — docs path sufficient at pin |
| EQOA-A7-030 | `CERT_CAMPAIGN_DELEGATED_DECISIONS_2026-11-07.md` (:211 and :216) records "re-audit authorised after correctives" and "audit obligation stays attached". It names no obligation owner, target SHA or date for DS-3, SSOT-D2, SSOT-D3, F-03 or F-04. Closure needs an owner/operator assignment. The slice does not touch it. | `DOC_PATCH` | eqo_closure/runs/*/COR_DRV.md cross-ref for EQOA-A7-030 | NONE — docs path sufficient at pin |

## Aggregate rationale (session E)

Session E aggregate **`FAIL`** — corrective product work is explicitly out of scope for wave 9; routes above are limited to qualification / governance documentation under `eqo_closure/` and linked owner records.

## DRV trigger (pin / train move)

- After #680 / train pin advances: execute `EQO-W3-WB01-DRV` (`RV-01`); do not open a new EQO campaign.
- Product fixes and pytest replays: wave 10+ lanes per `EQO_WAVE9_MANIFEST.md`.



## DRV disposition (wave 14, docs-only)

**Task ID:** `EQO-W3-WB01-DRV`
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
