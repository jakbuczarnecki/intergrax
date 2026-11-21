# EQO-W3-WB06 — COR/DRV disposition paths (wave 9)

**Task ID:** `EQO-W3-WB06-COR`
**COR outcome:** `NOT_APPLICABLE_WITH_EVIDENCE` for product CL-01 in this wave — documentation disposition paths only.
**AUD session E (aggregate):** `FAIL` (`runs/wb06/AUD_SESSION_E.md`)
**Evidence pin:** TRAIN_PIN `28c27cb896d02e0185b9fa02047411fa152f6469`
**Ref:** `governance/tickets/T-IGX-046-F.md`

## Source artefacts

| Artefact | Role |
|----------|------|
| `runs/wb06/AUD_SESSION_E.md` | Independent session E verdict (input) |
| `runs/wb06/W3_CLS.md` | Partial CLS pre-close (wave 5) |
| `runs/wb06/COR_DRV.md` | This wave-9 docs routing record |

## AUD packet binding

**Packet:** `AUD_PACKET.md` (`OPEN`, land SHA `8631b1e8459e096b74d1c80247d3e432b44fff30`)

## Finding disposition paths (docs-only @ pin)

| Finding / obligation | AUD verdict | Docs disposition path | Target artefact | Product / EXE route (deferred) |
|----------------------|-------------|----------------------|-----------------|-------------------------------|
| EQOA-A4-025 | The scale is confirmed by source: 27 `os.environ` / `os.getenv` read lines in 22 files under `intergrax/runtime/` (hypothesis: about 25). The CONFIG-X semantic scan recognises `environ.get` calls (`_config_x_semantic_production_scan.py:259-272`) but is driven by an inventory of concern surfaces, and `intergrax/runtime` appears there only for `observability/exporters/*` (`_config_x_concern_inventory.py:122`). Whether the scan is limited to inventoried surfaces, and whether the other reads are acceptable, needs the CONFIG-X gates to run, and none ran. It also needs an owner decision on the scope statement. | `DOC_PATCH` | eqo_closure/runs/*/COR_DRV.md cross-ref for EQOA-A4-025 | DEFERRED — product CL-01 / dedicated ticket (out of wave 9) |
| EQOA-A4-042 | Defect reproduced by reading source. `intergrax/globals/settings.py:34,37,40,43` set `default_language="pl"`, `default_locale="pl-PL"`, `default_region="pl-PL"`, `default_timezone="Europe/Warsaw"` as import-time class defaults (environment variables can override them). A search of `docs/project/architecture`, `docs/project/maintainers/plans`, `governance` and `scripts/ci` finds no reference to `intergrax/globals` or `intergrax.globals`, so no owner is recorded. Both parts of the hypothesis hold at the pin. No test in the slice covers it. Write set needs an owner assignment and a neutral-default decision. | `DOC_PATCH` | eqo_closure/ or owner qualification record | NONE — docs path sufficient at pin |
| EQOA-A5-002 | The stated gap is a cert-time gap that was later repaired, and the repair holds at the pin by source. Each named stream path now goes through the guarded `BaseLLMAdapter._execute_streaming` (`base_llm_adapter.py:380`, which calls `assert_protected_provider_call_allowed` and opens the stream through `provider_stream_admission`): Gemini (`gemini_adapter.py:231,271,453`), OpenAI Responses (`openai_responses_adapter.py:715,836`) and the LangChain Ollama adapter (`ollama_adapter.py:420`, built on `langchain_ollama`). Limit: this PASS rests on source reading only. The selected probe (`test_unresolved_capabilities_fail_closed`, present at `test_ollama_tool_calling.py:636`) did not run, and the historical cert-time gap itself is not reopened or erased. | `DOC_PATCH` | eqo_closure/runs/*/COR_DRV.md cross-ref for EQOA-A5-002 | DEFERRED — wave 10 EXE lane (out of wave 9) |

## Aggregate rationale (session E)

Session E aggregate **`FAIL`** — corrective product work is explicitly out of scope for wave 9; routes above are limited to qualification / governance documentation under `eqo_closure/` and linked owner records.

## DRV trigger (pin / train move)

- After #680 / train pin advances: execute `EQO-W3-WB06-DRV` (`RV-01`); do not open a new EQO campaign.
- Product fixes and pytest replays: wave 10+ lanes per `EQO_WAVE9_MANIFEST.md`.

## Limits and non-claims

- Derived from `AUD_SESSION_E.md` and existing DIS artefacts only; no pytest re-run in this lane.
- No G6 closure, IIAC IN_FORCE, FRZ promotion, or FCS readiness claim.
- No product code, scripts, tests, or register apply in this write set.
