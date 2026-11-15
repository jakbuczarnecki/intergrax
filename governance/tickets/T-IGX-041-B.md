# T-IGX-041-B — PDC-X batch B-P7-D2: repair of the IA-D2 audit findings for domain D2 (CONTEXT-KNOWLEDGE), loop 1

## Scope

Repair ticket for the independent audit `PDC_P7_D2_AUDIT_REPORT.md` (IA-D2, audited SHA `c08efe46a74a20595468d8d8bf06eab2dca5f573`, verdict PARTIAL), under Operator option B of `DELEGATION-PDC-P7-REPAIR.md`: real defects in existing code and test hygiene only. In scope: F-D2-01 (compiler, degradation ladder and mandatory-reserve half: the hard model-window budget counts assistant `tool_calls`, `name` and `tool_call_id`; the adapter estimator half is T-IGX-041-C and merges first), F-D2-17 (last-resort drop is oldest-first), F-D2-02 and F-D2-14 bounded (TASK scope of the memory control plane is tenant-qualified, bound-tenant guarded and governed; protocol-based guard), F-D2-11 isolation half (a raising reindex hook cannot break retrieval), F-D2-13 clock and deadline half (bounded `as_of`, bounded sweep run), F-D2-08 (the 35 red D2-cited RAG and context test files root-caused: stale doubles, removed APIs, missing extras, product bugs), F-D2-09 (syntax error and a syntax gate over the cited test trees), F-D2-10 (stale pre-context policy wiring gate and a docstring-coupled test), F-D2-18 (weak negative tests). Excluded with reason and reopen condition (plan section "Excluded findings"): F-D2-03 (TENANT-X carrier re-pin, T-IGX-041-I), F-D2-04, F-D2-05, F-D2-06 (F-04 family, hotspot composition), F-D2-07, F-D2-12, F-D2-15, F-D2-16, the tenant, payload-user and strict-profile halves of F-D2-11, F-D2-13 and F-D2-14, F-D2-19 and F-D2-20. Product paths: yes (`intergrax/runtime/nexus/context/`, `intergrax/context/budget/mandatory_base_messages.py`, `intergrax/memory/`, `intergrax/rag/governance/`, `intergrax/rag/retrieval/retrieval_service.py`, `intergrax/runtime/policy/pre_context_policy_audit.py`, one maintenance script). No hotspot path is written; `intergrax/llm_adapters/` is not touched. This ticket is not a certification verdict and not a closure of D2; the product-path changes keep the IIAC activation stale (ACT-N11) until the P8-04 rebind and the EBH-7 identity gate red until the final re-pin. Plan: `PLAN_T-IGX-041-B.md`.

## Plan digest

Plan digest: 8c15872d184a13636130e868d98e473fdead526a4edaedc65673600e2efc6f94

Canonical plan: [`governance/tickets/plans/PLAN_T-IGX-041-B.md`](plans/PLAN_T-IGX-041-B.md).

## Test contract

```text
uv run pytest tests/qualification/pdc_x/d2_repair tests/qualification/pdc_x/d2 tests/qualification/ce_01 tests/qualification/ce_02 tests/qualification/memory_behavior tests/unit/context tests/unit/memory tests/unit/runtime/nexus/context tests/unit/rag tests/integration/rag tests/integration/context tests/unit/runtime/architecture/test_ideal_harness_l3_w2_depth_gate.py -q --continue-on-collection-errors
uv run ruff check intergrax/runtime/nexus/context intergrax/context/budget/mandatory_base_messages.py intergrax/memory intergrax/rag/governance/embedding_version_policy.py intergrax/rag/retrieval/retrieval_service.py intergrax/runtime/policy/pre_context_policy_audit.py tests/qualification/pdc_x/d2_repair
uv run pyright intergrax/runtime/nexus/context/message_token_cost.py intergrax/runtime/nexus/context/context_compiler.py intergrax/runtime/nexus/context/degradation_ladder.py intergrax/memory/default_memory_control_plane.py intergrax/memory/retention_sweep.py tests/qualification/pdc_x/d2_repair
uv run python -m scripts.governance.igx_gov ticket
```

Plus the runtime mutation run (a)-(d) of plan Protocol item 4.

## Out of scope

- All excluded findings listed in the plan; composition of the retention sweep or the freshness observer; the formatter role contract (F-D2-04); the TENANT-X carrier re-pin (T-IGX-041-I); the adapter estimator half of F-D2-01 (T-IGX-041-C).
- `scripts/governance/`, `.github/`, `pyproject.toml`, `uv.lock`, `intergrax/contracts/`, every `intergrax/applications/_shared/` file, `intergrax/llm_adapters/`, the requirements matrix, the EBH-7 re-pin, the IIAC activation record and rebind, FRZ promotion, OD signatures, certification.

## Lifecycle state

plan_pending
