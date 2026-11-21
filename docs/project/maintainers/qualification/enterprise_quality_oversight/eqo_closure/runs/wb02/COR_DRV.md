# EQO-W3-WB02 — COR/DRV disposition paths (wave 9)

**Task ID:** `EQO-W3-WB02-COR`
**COR outcome:** `NOT_APPLICABLE_WITH_EVIDENCE` for product CL-01 in this wave — documentation disposition paths only.
**AUD session E (aggregate):** `FAIL` (`runs/wb02/AUD_SESSION_E.md`)
**Evidence pin:** TRAIN_PIN `28c27cb896d02e0185b9fa02047411fa152f6469`
**Ref:** `governance/tickets/T-IGX-046-F.md`

## Source artefacts

| Artefact | Role |
|----------|------|
| `runs/wb02/AUD_SESSION_E.md` | Independent session E verdict (input) |
| `runs/wb02/W3_CLS.md` | Partial CLS pre-close (wave 5) |
| `runs/wb02/COR_DRV.md` | This wave-9 docs routing record |

## AUD packet binding

**Packet:** `AUD_PACKET.md` (`PRE_AUD`, land SHA `8631b1e8459e096b74d1c80247d3e432b44fff30`)

## Finding disposition paths (docs-only @ pin)

| Finding / obligation | AUD verdict | Docs disposition path | Target artefact | Product / EXE route (deferred) |
|----------------------|-------------|----------------------|-----------------|-------------------------------|
| EQOA-A4-006 | Hypothesis confirmed. `scripts/maintenance/check_runtime_event_tenant_propagation.py:15` still lists `intergrax/agents/uaep.py`, which does not exist at the pin. The script is referenced only in `scripts/ci/script_paths.py:211`. No test in the slice invokes it, and no CI/test path does either. The slice (tool invoker, middleware, TENANT-X P0 fast) does not exercise UAEP event tenant propagation, so "limitation" is unsupported. Route: owner re-binds the gate path (as in T-IGX-041-B), then replay. | `DOC_PATCH` | eqo_closure/runs/*/COR_DRV.md cross-ref for EQOA-A4-006 | DEFERRED — product CL-01 / dedicated ticket (out of wave 9) |
| EQOA-A4-027 | Defect reproduced by reading source. `applications/research_application/serving/fastapi_router.py:46-54` takes `tenant_id = body.tenant_id` when no authenticated principal is present. `applications/research_application/serving/schemas.py:9` defaults `tenant_id: str = "research-tenant"`. A missing tenant silently becomes a fixed default, which is the FRZ-TEN-10 failure mode while FRZ-TEN-10 is recorded PASS. No test in the slice covers this. Write set needs OD-EQO-08 (serving-router ticket). | `OWNER_DECISION_RECORD` | eqo_closure/w4/W4-03_OWNER_DECISION_BATCH.md or operator packet | NONE — docs path sufficient at pin |
| EQOA-A7-009 | Label drift confirmed: `EBH_7_GAP_REGISTER.md:5` says `STRUCTURAL — N/A with evidence (not E2E)`. `EBH_7_CLOSURE_EVIDENCE_AUDIT_MATRIX.md:141` (G09) says `PASS` with the note "Structural N/A evidence only — not platform E2E proof". No pytest in the slice replays E2E for the five `unknown_role` surfaces. Closure needs the owner/operator accepted-limitation record (owner, scope, reopen condition; EDER-D-0646 `ACCEPTED_AS_STATED_RECORD_REQUIRED`) plus label standardisation. | `ACCEPTED_SCOPED_LIMITATION_RECORD` | qualification stage record + EDER cross-ref | DEFERRED — wave 10 EXE lane (out of wave 9) |
| EQOA-A7-028 | EBH-7 is `CLOSED — INDEPENDENTLY ACCEPTED (SCOPED)` (OD-EBH7-08); the production-delta residual is not replayed at the pin (manifest blocker `EBH7_TENANT_E2E_NOT_REPLAYED`). The slice cannot decide it; it needs an EBH-7 replay or an operator disposition. | `OWNER_DECISION_RECORD` | eqo_closure/w4/W4-03_OWNER_DECISION_BATCH.md or operator packet | DEFERRED — product CL-01 / dedicated ticket (out of wave 9) |
| EQOA-A7-031 | Agree with EXE. The skips are `pytest.skip("... run in tenant_platform_e2e module" / "... dedicated fixture test")` in `tests/qualification/ebh_7/test_ebh_7_w6_tenant_e2e_behavioral_gates.py:27,29`. No routed-proof replay was executed. | `DOC_PATCH` | eqo_closure/runs/*/COR_DRV.md cross-ref for EQOA-A7-031 | DEFERRED — wave 10 EXE lane (out of wave 9) |
| EQOA-A8-006 | Gap confirmed: 24 of 73 `docs/project/architecture/*.md` contain no occurrence of "tenant". The finding is LOW and NON_BLOCKING_WITH_EVIDENCE. The DIS honestly says pytest cannot close it. Closure needs an owner doc patch (tenant/security section or pointer to the TENANT-X record) or an acceptance record. | `DOC_PATCH` | eqo_closure/ or owner qualification record | DEFERRED — wave 10 EXE lane (out of wave 9) |

## Aggregate rationale (session E)

Session E aggregate **`FAIL`** — corrective product work is explicitly out of scope for wave 9; routes above are limited to qualification / governance documentation under `eqo_closure/` and linked owner records.

## DRV trigger (pin / train move)

- After #680 / train pin advances: execute `EQO-W3-WB02-DRV` (`RV-01`); do not open a new EQO campaign.
- Product fixes and pytest replays: wave 10+ lanes per `EQO_WAVE9_MANIFEST.md`.



## DRV disposition (wave 14, docs-only)

**Task ID:** `EQO-W3-WB02-DRV`
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
