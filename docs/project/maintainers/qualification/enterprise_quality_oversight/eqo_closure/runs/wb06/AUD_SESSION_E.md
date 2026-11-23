# EQO-W3-WB06-AUD — session E verdict (wave AUD-2)

**Verdict (WB06): `FAIL`**
**Auditor session:** independent session E (not the EXE agent; no EXE/DIS artefact in this WB authored by this session).
**Date:** 2026-11-21
**Evidence pin:** TRAIN_PIN `28c27cb896d02e0185b9fa02047411fa152f6469` (source read at the pin, read-only)
**Packet:** `AUD_PACKET.md` (`OPEN`, land SHA `8631b1e8459e096b74d1c80247d3e432b44fff30`)
**Ref:** `governance/tickets/T-IGX-046-F.md`

## Verdict vocabulary

| Verdict | Meaning |
|---------|---------|
| `PASS` | Evidence at the pin supports the stated disposition. |
| `FAIL` | Source at the pin confirms the defect or contradicts the stated disposition. The finding stays OPEN. |
| `BLOCKED` | The pytest slice does not produce evidence to decide, or an owner/operator decision is a prerequisite. |

Aggregate rule: any `FAIL` gives `FAIL`; otherwise any `BLOCKED` gives `BLOCKED`; otherwise `PASS`.

## Packet checks

| Check | Result |
|-------|--------|
| TRAIN_PIN in manifest / DES_PREREG / packet = `28c27cb8…` | YES (object exists locally) |
| `pytest_command_sha256` recomputed from `pytest_command` (UTF-8) | MATCH `51d8f0b8…` (manifest = DES_PREREG) |
| Log vs manifest summary | MATCH: 1 error in 0.88s (exit 4) |
| Log records command, node ids or its own hash | NO (summary lines only; log hash not verifiable) |
| TRAIN_PIN is an ancestor of `origin/development` | NO (it sits on the `af2/*` train). Verdicts bind to the pin, not to `development` HEAD. |

**What the slice executes:** nothing. `tests/unit/llm_adapters/test_ollama_tool_calling.py` failed at import (`ModuleNotFoundError: No module named 'langchain_core'`, an optional dependency of the `llm-langchain-ollama` extra). Pytest then reported "found no collectors" for the selected node and exited 4. The two CONFIG-X files in the same command produced no result. The EXE exit 4 is an environment gap, not a product verdict. Zero tests ran.

## Per-finding verdicts

| Finding | Interim DIS | AUD verdict | Rationale (evidence at pin) |
|---------|-------------|-------------|-----------------------------|
| EQOA-A4-025 | BLOCKED | **BLOCKED** | The scale is confirmed by source: 27 `os.environ` / `os.getenv` read lines in 22 files under `intergrax/runtime/` (hypothesis: about 25). The CONFIG-X semantic scan recognises `environ.get` calls (`_config_x_semantic_production_scan.py:259-272`) but is driven by an inventory of concern surfaces, and `intergrax/runtime` appears there only for `observability/exporters/*` (`_config_x_concern_inventory.py:122`). Whether the scan is limited to inventoried surfaces, and whether the other reads are acceptable, needs the CONFIG-X gates to run, and none ran. It also needs an owner decision on the scope statement. |
| EQOA-A4-042 | BLOCKED | **FAIL** | Defect reproduced by reading source. `intergrax/globals/settings.py:34,37,40,43` set `default_language="pl"`, `default_locale="pl-PL"`, `default_region="pl-PL"`, `default_timezone="Europe/Warsaw"` as import-time class defaults (environment variables can override them). A search of `docs/project/architecture`, `docs/project/maintainers/plans`, `governance` and `scripts/ci` finds no reference to `intergrax/globals` or `intergrax.globals`, so no owner is recorded. Both parts of the hypothesis hold at the pin. No test in the slice covers it. Write set needs an owner assignment and a neutral-default decision. |
| EQOA-A5-002 | BLOCKED | **PASS** | The stated gap is a cert-time gap that was later repaired, and the repair holds at the pin by source. Each named stream path now goes through the guarded `BaseLLMAdapter._execute_streaming` (`base_llm_adapter.py:380`, which calls `assert_protected_provider_call_allowed` and opens the stream through `provider_stream_admission`): Gemini (`gemini_adapter.py:231,271,453`), OpenAI Responses (`openai_responses_adapter.py:715,836`) and the LangChain Ollama adapter (`ollama_adapter.py:420`, built on `langchain_ollama`). Limit: this PASS rests on source reading only. The selected probe (`test_unresolved_capabilities_fail_closed`, present at `test_ollama_tool_calling.py:636`) did not run, and the historical cert-time gap itself is not reopened or erased. |

## Rationale for the aggregate

`FAIL` because A4-042 is confirmed by source at the pin and the slice gives it no coverage. A4-025 is not decidable without a working slice. A5-002 is supported by source.

## Required follow-up (no product fix here)

1. Re-run the slice with the `llm-langchain-ollama` extra installed, or split the unit probe from the CONFIG-X files so one missing extra cannot hide them. Attach the log.
2. A4-042: assign an owner for `intergrax/globals` and decide whether the Poland-specific defaults stay, move to configuration or become neutral.
3. A4-025: replay the CONFIG-X gates and record which of the 22 runtime files are inside the closed-world scan, or record the scope limitation with an owner.
4. A5-002: attach the replay of the guard probe when the extra is available (confirmation only).

## Limits and non-claims

- This verdict is based on the AUD packet, the files it lists, and read-only inspection of the source at TRAIN_PIN. Pytest was not re-executed by this session.
- This is a per-WB closure-evidence verdict only. It makes no claim of G6 closure, IIAC, certification, FRZ promotion or FCS readiness, and it does not set `IIAC_POLICY_ACTIVATION_RECORD.json`.
- No product code was changed.
