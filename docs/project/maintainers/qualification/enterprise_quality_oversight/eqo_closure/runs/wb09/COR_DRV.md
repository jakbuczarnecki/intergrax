# EQO-W3-WB09 — COR/DRV disposition paths (wave 9)

**Task ID:** `EQO-W3-WB09-COR`
**COR outcome:** `NOT_APPLICABLE_WITH_EVIDENCE` for product CL-01 in this wave — documentation disposition paths only.
**AUD session E (aggregate):** `FAIL` (`runs/wb09/AUD_SESSION_E.md`)
**Evidence pin:** TRAIN_PIN `28c27cb896d02e0185b9fa02047411fa152f6469`
**Ref:** `governance/tickets/T-IGX-046-F.md`

## Source artefacts

| Artefact | Role |
|----------|------|
| `runs/wb09/AUD_SESSION_E.md` | Independent session E verdict (input) |
| `runs/wb09/W3_CLS.md` | Partial CLS pre-close (wave 5) |
| `runs/wb09/COR_DRV.md` | This wave-9 docs routing record |

## AUD packet binding

**Packet:** `AUD_PACKET.md` (`OPEN`, land SHA `8631b1e8459e096b74d1c80247d3e432b44fff30`)

## Finding disposition paths (docs-only @ pin)

| Finding / obligation | AUD verdict | Docs disposition path | Target artefact | Product / EXE route (deferred) |
|----------------------|-------------|----------------------|-----------------|-------------------------------|
| EQOA-A2-027 | Stale scope note confirmed in source. `intergrax/skills/host_lifecycle.py` defines `disable_skill` (:112), `remove_skill` (:149) and `upgrade_skill` (:184). `PLATFORM_ENTERPRISE_COMPLETION_ROADMAP.md:487` still carries the SKILL-X scope note that disable/remove/upgrade are unsupported and untested (`CERT_CAMPAIGN_DELEGATED_DECISIONS_2026-11-07.md:149,189`, F-04). Acceptance stays bound to `552e3798`, not to the changed code, and the slice does not exercise those methods. The PROD-Q half of the hypothesis is not reproduced: the strict-serving wiring gate passes at the pin, so that regression is no longer present, but that does not close the stale scope note or the requalification of the changed lifecycle surface. Route: owner updates the scope note and requalifies, or states the surface is outside the accepted scope. | `DOC_PATCH` | eqo_closure/ or owner qualification record | DEFERRED — wave 10 EXE lane (out of wave 9) |
| EQOA-A4-040 | Defect reproduced by reading source. Direct `uvicorn.run` serving entrypoints at the pin: `applications/attestation_demo/host/main.py:19`, `applications/intergrax_assistant_application/host/main.py:19`, `applications/lab_application/host/main.py:22`, `applications/poc_template_application/host/main.py:19`, `applications/research_application/host/main.py:98` and `applications/governed_contractor_application/host/package_b_strict_production_serving_command.py:159`. Only `applications/local_workspace_application` imports the hosting lifecycle engine (`intergrax.hosting`). "Most Tier-3 entrypoints" holds. The strict-serving gate that passes covers the governed contractor package-B command, not the other five. The P5 wrapper failure is not related. Write set needs an operator decision on the hosting-engine adoption scope for reference apps. | `OWNER_DECISION_RECORD` | eqo_closure/w4/W4-03_OWNER_DECISION_BATCH.md or operator packet | DEFERRED — product CL-01 / dedicated ticket (out of wave 9) |

## Aggregate rationale (session E)

Session E aggregate **`FAIL`** — corrective product work is explicitly out of scope for wave 9; routes above are limited to qualification / governance documentation under `eqo_closure/` and linked owner records.

## DRV trigger (pin / train move)

- After #680 / train pin advances: execute `EQO-W3-WB09-DRV` (`RV-01`); do not open a new EQO campaign.
- Product fixes and pytest replays: wave 10+ lanes per `EQO_WAVE9_MANIFEST.md`.



## DRV disposition (wave 14, docs-only)

**Task ID:** `EQO-W3-WB09-DRV`
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
