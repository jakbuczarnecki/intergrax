# EQO-W3-WB12 — COR/DRV disposition paths (wave 9)

**Task ID:** `EQO-W3-WB12-COR`
**COR outcome:** `NOT_APPLICABLE_WITH_EVIDENCE` for product CL-01 in this wave — documentation disposition paths only.
**AUD session E (aggregate):** `BLOCKED` (`runs/wb12/AUD_SESSION_E.md`)
**Evidence pin:** TRAIN_PIN `28c27cb896d02e0185b9fa02047411fa152f6469`
**Ref:** `governance/tickets/T-IGX-046-F.md`

## Source artefacts

| Artefact | Role |
|----------|------|
| `runs/wb12/AUD_SESSION_E.md` | Independent session E verdict (input) |
| `runs/wb12/W3_CLS.md` | Partial CLS pre-close (wave 5) |
| `runs/wb12/COR_DRV.md` | This wave-9 docs routing record |

## AUD packet binding

**Packet:** `AUD_PACKET.md` (`PRE_AUD`, land SHA `8631b1e8459e096b74d1c80247d3e432b44fff30`)

## Finding disposition paths (docs-only @ pin)

| Finding / obligation | AUD verdict | Docs disposition path | Target artefact | Product / EXE route (deferred) |
|----------------------|-------------|----------------------|-----------------|-------------------------------|
| EQOA-A3-019 | The hypothesis is that row-level PASS evidence is tied to certification-time SHAs while production code changed later. The slice has no SHA-versus-delta check. The packet does not cite a replay of EXE/GOV/CTL/REC/HRN rows at a current SHA, so the claim is neither confirmed nor refuted. | `DOC_PATCH` | eqo_closure/runs/*/COR_DRV.md cross-ref for EQOA-A3-019 | DEFERRED — product CL-01 / dedicated ticket (out of wave 9) |
| EQOA-A4-039 | Duplication confirmed at the pin: two `ProofReceipt` models (`intergrax/contracts/execution_evidence/receipt.py:26` and `intergrax/proofs/receipts/contracts.py:30`), plus `ProofReceiptStore` as a class (`intergrax/proofs/receipts/store.py:19`) and as a Protocol (`applications/governed_contractor_application/host/stores.py:83`). The slice tests only `intergrax.proofs.receipts`, so it cannot show the models are safely separated. Acceptance needs an owner decision (unify, rename, or sanction) with a recorded scope. LOW severity. | `DOC_PATCH` | eqo_closure/runs/*/COR_DRV.md cross-ref for EQOA-A4-039 | DEFERRED — product CL-01 / dedicated ticket (out of wave 9) |
| EQOA-A4-047 | The semantic duplicate-mechanism gate (ENT-A-DUP-01) was skipped, and the four-mechanism scenario overlap and the SFX closed-world resolver mapping have no test in the slice. The master matrix rates its freeze impact NON_BLOCKING_WITH_EVIDENCE, but that evidence is absent from this packet. | `DOC_PATCH` | docs/project/architecture/ owner mapping memo | DEFERRED — wave 10 EXE lane (out of wave 9) |
| EQOA-A5-010 | The question is whether 70 product modules added since the acceptance pin were re-examined. The slice's closed-world checks count catalogs and registries. They do not diff modules against the acceptance pin, and the semantic gate was skipped. No evidence about the added modules is in the packet. | `DOC_PATCH` | eqo_closure/runs/*/COR_DRV.md cross-ref for EQOA-A5-010 | DEFERRED — product CL-01 / dedicated ticket (out of wave 9) |
| EQOA-A8-020 | Not decidable by pytest. The Protocol-v2 remediation (LCI trusted tenant scope, Token Optimization receipt identity) appears only in the EQO dossiers and the debt register. `PLATFORM_ENTERPRISE_COMPLETION_ROADMAP.md` has no "receipt identity" entry at the pin, and no roadmap or FRZ tracking row was found. Closure needs an owner/operator decision to track it (roadmap or FRZ row) or to except it. | `DOC_PATCH` | eqo_closure/ or owner qualification record | DEFERRED — wave 10 EXE lane (out of wave 9) |
| EQOA-A2-031 | The NPSC-4.2 allowlist ("FAIL retained" text versus the allowlist gate refreshed under EBH-7 W3) is not exercised by any test in the slice. The packet has no recorded result for that gate. | `DOC_PATCH` | eqo_closure/runs/*/COR_DRV.md cross-ref for EQOA-A2-031 | NONE — docs path sufficient at pin |

## Aggregate rationale (session E)

Session E aggregate **`BLOCKED`** — corrective product work is explicitly out of scope for wave 9; routes above are limited to qualification / governance documentation under `eqo_closure/` and linked owner records.

## DRV trigger (pin / train move)

- After #680 / train pin advances: execute `EQO-W3-WB12-DRV` (`RV-01`); do not open a new EQO campaign.
- Product fixes and pytest replays: wave 10+ lanes per `EQO_WAVE9_MANIFEST.md`.



## DRV disposition (wave 14, docs-only)

**Task ID:** `EQO-W3-WB12-DRV`
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
