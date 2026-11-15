# T-IGX-041-C — PDC-X batch B-P7-D3: repair of the independent D3 (MODEL-RUNTIME) audit findings, loop 1

## Scope

Product and test repair after the independent audit IA-D3 (`PDC_P7_D3_AUDIT_REPORT.md`, verdict PARTIAL, audited SHA `c08efe46a74a20595468d8d8bf06eab2dca5f573`), under the Operator's option B (`DELEGATION-PDC-P7-REPAIR.md`): real defects in existing code and test hygiene only. In scope: F-D3-01 (stream paths of the default OpenAI, Gemini/Vertex and LangChain Ollama adapters go through `_execute_streaming`: guard, quota, resilience, cancel-transport registration), F-D3-02 (refusal and content-filter normalization on every stream and buffered path), F-D3-05 (stale and state-dependent `llm_adapters` unit suites, the fixture agent that cannot be instantiated), the substring-classification defect of F-D3-07 (word-boundary retry classification), the doc part of F-D3-03 (`timeout_sec` is an admission budget), the adapter-estimator half of D2 finding F-D2-01 (`tool_calls` counted), and by-products F-D3-08 and F-D3-13. Excluded with reason and reopen condition (plan, table "Excluded findings"): F-D3-03 enforcement, F-D3-04 (resolver wiring, F-04 family), F-D3-06 (ticket T-IGX-041-I), the typed taxonomy of F-D3-07, and all LOW/INFO findings. Product code: yes (`intergrax/llm_adapters/` and one `platform_proofs/` scenario agent). No hotspot path is written. This ticket is not a certification verdict; IIAC activation stays stale until the P8-04 rebind and the EBH-7 identity gate stays red until the final re-pin. Plan: `PLAN_T-IGX-041-C.md`.

## Plan digest

Plan digest: a42ea80029a3e8af786ac6d909715c260205b0a6f8d11af6a1d0cfc6e3bbe5a3

Canonical plan: [`governance/tickets/plans/PLAN_T-IGX-041-C.md`](plans/PLAN_T-IGX-041-C.md).

## Approved

(pending; the plan is signed by a delegate session under `DELEGATION-PDC-P7-REPAIR.md` or by the Operator; the implementer never writes the approval line)

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

plan_pending
