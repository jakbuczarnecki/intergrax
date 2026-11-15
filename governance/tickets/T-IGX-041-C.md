# T-IGX-041-C — PDC-X batch B-P7-D3: repair of the independent D3 (MODEL-RUNTIME) audit findings, loop 1

## Scope

Product and test repair after the independent audit IA-D3 (`PDC_P7_D3_AUDIT_REPORT.md`, verdict PARTIAL, audited SHA `c08efe46a74a20595468d8d8bf06eab2dca5f573`), under the Operator's option B (`DELEGATION-PDC-P7-REPAIR.md`): real defects in existing code and test hygiene only. In scope: F-D3-01 (stream paths of the default OpenAI, Gemini/Vertex and LangChain Ollama adapters go through `_execute_streaming`: guard, quota, resilience, cancel-transport registration), F-D3-02 (refusal and content-filter normalization on every stream and buffered path), F-D3-05 (stale and state-dependent `llm_adapters` unit suites, the fixture agent that cannot be instantiated), the substring-classification defect of F-D3-07 (word-boundary retry classification), the doc part of F-D3-03 (`timeout_sec` is an admission budget), the adapter-estimator half of D2 finding F-D2-01 (`tool_calls` counted), and by-products F-D3-08 and F-D3-13. Excluded with reason and reopen condition (plan, table "Excluded findings"): F-D3-03 enforcement, F-D3-04 (resolver wiring, F-04 family), F-D3-06 (ticket T-IGX-041-I), the typed taxonomy of F-D3-07, and all LOW/INFO findings. Product code: yes (`intergrax/llm_adapters/` and one `platform_proofs/` scenario agent). No hotspot path is written. This ticket is not a certification verdict; IIAC activation stays stale until the P8-04 rebind and the EBH-7 identity gate stays red until the final re-pin. Plan: `PLAN_T-IGX-041-C.md`.

## Plan digest

Plan digest: a42ea80029a3e8af786ac6d909715c260205b0a6f8d11af6a1d0cfc6e3bbe5a3

Canonical plan: [`governance/tickets/plans/PLAN_T-IGX-041-C.md`](plans/PLAN_T-IGX-041-C.md).

## Approved

Approved: operator-delegate @ 2026-11-15T08:57:00+01:00
Delegation: docs/project/maintainers/qualification/platform_domain_completeness/operator_decisions/DELEGATION-PDC-P7-REPAIR.md@8f4b139c1f5352c9861872f0b5077996b8f0f891

Delegate checks (scope of `DELEGATION-PDC-P7-REPAIR.md`, all passed before this line was written; delegation read on `origin/development`, last changed by `8f4b139c1`; plan read at the batch branch, commit `93972680d`, base `1c864ea19`):

- The plan digest recomputed with `scripts/governance/plan_digest.py` equals the digest above (`a42ea80029a3e8af786ac6d909715c260205b0a6f8d11af6a1d0cfc6e3bbe5a3`).
- The product and test write set of the plan (24 entries) equals the write set of batch row `B-P7-D3` in `PDC_IMPLEMENTATION_WBS.md` (compared mechanically: no entry on either side only; the plan's only extra backticked name, `test_d3_r022_capability_flags.py`, lies inside the listed directory `tests/qualification/pdc_x/d3/`). The commit diff against `1c864ea19` touches only the ticket, the plan, the WBS, the DAG and `scripts/qualification/pdc_x_wbs.py`; no product path changed.
- Hotspots: nothing under `intergrax/applications/_shared/` or `intergrax/contracts/` is in the write set; both are listed as out of scope and F-D3-04 is excluded because it would touch `intergrax/applications/_shared/llm_resolver.py`. The write-set paths `intergrax/llm_adapters/_shared/...` and `intergrax/llm_adapters/contracts/call_config.py` are different directories (the latter exists) and are named exactly, with one new file `stream_finish_signal.py`. The only directory entry is a test directory, `tests/qualification/pdc_x/d3/`.
- Findings: in-scope table (F-D3-01, -02, -05, F-D3-07 defect part, F-D3-03 doc part, F-D2-01 estimator half, F-D3-08, F-D3-13) and excluded table (F-D3-03 enforcement and F-D3-07 taxonomy = missing feature versus claim, F-D3-04 = F-04 family, F-D3-06 -> T-IGX-041-I, LOW and INFO -> debt register), each exclusion with reason and reopen condition.
- F-D2-01 estimator half: the plan limits it to `estimate_tokens_for_messages` in `intergrax/llm_adapters/base/base_llm_adapter.py` (counts serialized `tool_calls`, `name`, `tool_call_id`; messages without them estimate as before, frozen table) and leaves the compiler, ladder and trim half under `intergrax/runtime/nexus/context/` to T-IGX-041-B. The D2 ticket is not yet written (branch `claude/pdc-x-041b-ad8a21` carries no ticket or plan); the tickets of A, E and G name no `intergrax/llm_adapters/` path. No overlap can be shown today; the D2 plan must not name `base_llm_adapter.py` (a signer of the D2 ticket should check that).
- Gates: no gate is weakened or deleted. `_declares_true` in `test_d3_r022_capability_flags.py` currently accepts a constant `True` or any `ast.Call` return (line 217-223); the plan tightens it to a constant `True` or a delegation to a named flag, with a negative test (`return bool(self._x)` must fail). If a shipped adapter fails the tighter predicate, the plan allows a named, commented allowlist entry for that adapter only, which is no weaker than the present predicate, and requires a report. Other edits of existing tests (`test_ollama_tool_calling.py` message regex, split of `test_invalid_adapter_missing_model_fails_clearly`, state isolation, `importorskip("anthropic")` for two modules) are justified in the plan as stale expectation or classification, with the Test contract running the modules unchanged with the llm extras; no test is deleted, xfailed or narrowed. A new source gate (AST) and pinned tests for mutants M03 and M12 strengthen.
- No registry, scheduler, engine, trust, policy or audit authority is added (design constraints of the plan); nothing under `scripts/governance/` or `.github/` is in the write set; the C10/C15 freeze is respected (no credential, endpoint or vendor addition).
- The plan states "This plan is not a certification verdict"; the implementer (branch `claude/pdc-x-041c-e320dd`) and this delegate session are different sessions. The delegate edited nothing except this approval block and the lifecycle state. Not checked by the delegate: the correctness of the plan's code-level claims (stream paths, line numbers), which the implementer re-verifies as the plan's first protocol step.
- Merge gate (stated as the delegation record requires): product pull requests merge under authorizations 1 and 3 of `DELEGATION-PDC-STANDING.md` and decision 8 of `DELEGATION-PDC-WAVE2.md`, as `DELEGATION-PDC-P7-REPAIR.md` names them; WBS merge gate OD-PDC-SEQ-01 (merge only after MCP-SERVER-X P8 publication) also applies. The delegate did not verify that the numbering matches the numbered items of `DELEGATION-PDC-STANDING.md`. Write-set amendments are not delegated.

## Test contract

```text
uv run --extra llm-openai --extra llm-anthropic --extra llm-mistral --extra llm-ollama --extra llm-langchain-ollama --extra llm-gemini --extra llm-bedrock --extra llm-cohere-native --extra integrations-duckdb pytest tests/unit/llm_adapters tests/qualification/ce_02 tests/qualification/pdc_x/d3 tests/qualification/pdc_x/d2 tests/qualification/pdc_x/p6_model tests/qualification/pdc_x/xd/test_xd_01_agent_model.py tests/qualification/ebh_5 tests/unit/platform_proofs/scenarios/ai_incident_investigation tests/unit/runtime/nexus/context -q -p no:randomly
uv run ruff check <touched files>
uv run pyright <touched files>
uv run python scripts/qualification/pdc_x_wbs.py --check
uv run python scripts/qualification/pdc_x_requirements.py --check
uv run python -m scripts.governance.igx_gov ticket
```

Plus the mutation run for M03 and M12 described in the plan (Protocol item 4).

## Out of scope

- The excluded findings (plan table), the TENANT-X carrier re-pin (T-IGX-041-I), the D2 compiler/ladder/trim half of F-D2-01 (T-IGX-041-B).
- `intergrax/applications/_shared/`, `intergrax/contracts/`, `intergrax/runtime/`, `scripts/governance/`, `.github/`, the failover registry, the model catalog, the quota module, FRZ files, P8 originals.
- IIAC activation and rebind, the EBH-7 re-pin, OD signatures, certification and the re-audit.

## Lifecycle state

plan_approved
