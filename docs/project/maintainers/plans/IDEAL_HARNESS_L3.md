# Ideal Harness L3 - Implementation Plan

**Architecture target:** [`guides/IDEAL_HARNESS_AI_ARCHITECTURE.md`](../../technical/guides/IDEAL_HARNESS_AI_ARCHITECTURE.md) §12.3
**Audit map:** [`../../../audit_results/AUDIT_PROTOCOL.md`](../../../audit_results/AUDIT_PROTOCOL.md) §8 (32 layers)
**Hub:** [`plan/PLATFORM_FOUNDATION.md`](PLATFORM_FOUNDATION.md) · Band **2ax** · queue **§6.1at**  
**Status:** **W2 Done** (2026-06-09) - P0+P1 harness depth closed; scorecard **32/32 L3**; Band 3 rows remain deferred

> **Scope:** Harness infrastructure only. Band 3 product rows (K.1/K.2, FLOW-8 product, CFG-14, GOV-PROD.1) remain in [§6.3](PLATFORM_FOUNDATION.md#63-end-of-plan--deferred-product-work-only).

---

## Delivery rule

One **IDEAL-* ID** per PR → update this register + affected domain plan row → `pytest -m gate` + `scripts/gates/check_ideal_harness_l3_gates.py` green.

---

## Phase waves

| Wave | Focus | Layer IDs |
|------|-------|-----------|
| **W0** | Plan + gates scaffold | All |
| **W1** | P0 critical L3 (identity, reliability, security, cost, ops) | §4, §10, §17, §20–§24, §31 |
| **W2** | P1 governance depth | §3, §5–§6, §11, §14–§19, §25–§28, §30, §32 |
| **W3** | P2 selective L4 hooks | §6, §24, §25 |
| **W4** | Product (Band 3) | §28 product rows |

---

## Master register

| ID | Layer | AUDIT § | Priority | Deliverable | Status |
|----|-------|---------|----------|-------------|--------|
| IDEAL-1.1 | Strategic Harness Model | 1 | P2 | `scripts/gates/harness_maturity_report.py` - 32-layer scorecard | **Done** |
| IDEAL-1.2 | Strategic Harness Model | 1 | P3 | Quarterly strategy review process (docs only) | **Done** |
| IDEAL-2.1 | Tier boundaries | 2 | P2 | `check_agents_no_tier3_imports.py` (existing) - extend agents→apps gate | **Done** |
| IDEAL-2.2 | Tier boundaries | 2 | P1 | `scripts/maintenance/check_agents_no_vendor_sdk_imports.py` | **Done** |
| IDEAL-3.1 | Task intake | 3 | P1 | Intake parity gate in `test_ideal_harness_l3_depth_gate.py` | **Done** |
| IDEAL-3.2 | Task intake | 3 | P1 | SLA/risk metadata validation on `TaskEnvelope` | **Done** |
| IDEAL-3.3 | Task intake | 3 | P2 | Streaming intake contract | **Done** |
| IDEAL-3.4 | Task intake | 3 | P2 | Durable async task index (product hosts) | **Done** |
| IDEAL-4.1 | Identity & trust | 4 | P0 | `actor_resolution.py` - user/service/agent from envelope | **Done** |
| IDEAL-4.2 | Identity & trust | 4 | P0 | Delegation scope narrowing in `actor_resolution.py` | **Done** |
| IDEAL-4.3 | Identity & trust | 4 | P0 | Tenant isolation gate tests | **Done** |
| IDEAL-4.4 | Identity & trust | 4 | P1 | Secrets rotation via IntegrationProfile hooks | **Done** |
| IDEAL-4.5 | Identity & trust | 4 | P1 | `DELEGATION_GRANTED` trace payload | **Done** |
| IDEAL-4.6 | Identity & trust | 4 | P2 | Impersonation rationale logging | **Done** |
| IDEAL-5.1 | Policy | 5 | P1 | Pre-context policy hook wiring audit | **Done** |
| IDEAL-5.2 | Policy | 5 | P1 | `policy_bundle_version` on run trace | **Done** |
| IDEAL-5.3 | Policy | 5 | P1 | HITL ↔ `PolicyDecision` correlation | **Done** |
| IDEAL-5.4 | Policy | 5 | P2 | Execution mode test matrix | **Done** |
| IDEAL-5.5 | Policy | 5 | P1 | Adversarial policy regression fixtures | **Done** |
| IDEAL-6.1 | LLM adapters | 6 | P1 | `check_agents_llm_adapter_response.py` (existing) enforce | **Done** |
| IDEAL-6.2 | LLM adapters | 6 | P1 | Fallback chain in `model_router.py` depth test | **Done** |
| IDEAL-6.3 | LLM adapters | 6 | P2 | W-ADAPT cost/latency routing integration | **Done** |
| IDEAL-6.4 | LLM adapters | 6 | P2 | Structured output validation gate | **Done** |
| IDEAL-7.1 | Cognition | 7 | P1 | `DecisionRecord` on all reference agents | **Done** |
| IDEAL-7.2 | Cognition | 7 | P2 | Reasoning failure taxonomy | **Done** |
| IDEAL-7.3 | Cognition | 7 | P2 | Prompt compilation layer audit | **Done** |
| IDEAL-8.1 | Execution runtime | 8 | P2 | Long-running resume E2E | **Done** |
| IDEAL-8.2 | Execution runtime | 8 | P1 | Idempotency on side-effect tools gate | **Done** |
| IDEAL-9.1 | Orchestration | 9 | P2 | Production queue adapter | **Done** |
| IDEAL-9.2 | Orchestration | 9 | P2 | Merge strategy test matrix | **Done** |
| IDEAL-9.3 | Orchestration | 9 | P2 | Provider degradation simulation | **Done** |
| IDEAL-10.1 | Subagents | 10 | P0 | `SubtaskContract` budget fields enforced (existing) | **Done** |
| IDEAL-10.2 | Subagents | 10 | P0 | Memory namespace isolation gate | **Done** |
| IDEAL-10.3 | Subagents | 10 | P1 | Tool allowlist per contract runtime enforce | **Done** |
| IDEAL-10.4 | Subagents | 10 | P1 | `inherit_tool_policy=False` default (existing) | **Done** |
| IDEAL-10.5 | Subagents | 10 | P1 | Delegation rationale in `DecisionRecord` | **Done** |
| IDEAL-10.6 | Subagents | 10 | P2 | Evaluator loop standard graph node | **Done** |
| IDEAL-11.1 | Tools | 11 | P1 | Shipped tool contract test gate | **Done** |
| IDEAL-11.2 | Tools | 11 | P2 | Oversized tool lint | **Done** |
| IDEAL-11.3 | Tools | 11 | P1 | HIGH risk → HITL policy test | **Done** |
| IDEAL-12.1 | Skills | 12 | P2 | SkillImporter fuzz tests | **Done** |
| IDEAL-12.2 | Skills | 12 | P2 | Skill dependency graph CI | **Done** |
| IDEAL-13.1 | Integrations | 13 | P2 | Core provider weekly smoke | **Done** |
| IDEAL-13.2 | Integrations | 13 | P2 | Rate-limit conformance per category | **Done** |
| IDEAL-14.1 | RAG | 14 | P1 | Golden retrieval regression dataset | **Done** |
| IDEAL-14.2 | RAG | 14 | P1 | Citation preservation contract test | **Done** |
| IDEAL-14.3 | RAG | 14 | P1 | Retrieval poisoning adversarial suite | **Done** |
| IDEAL-14.4 | RAG | 14 | P2 | Graph RAG production profile | **Done** |
| IDEAL-15.1 | Memory | 15 | P2 | Procedural memory store | **Done** |
| IDEAL-15.2 | Memory | 15 | P1 | Forget/delete E2E with retention audit | **Done** |
| IDEAL-15.3 | Memory | 15 | P2 | Memory provenance on context fragments | **Done** |
| IDEAL-16.1 | Context | 16 | P1 | Context regression golden suite | **Done** |
| IDEAL-16.2 | Context | 16 | P2 | Semantic compression strategy | **Done** |
| IDEAL-16.3 | Context | 16 | P1 | Citation chain output→fragment→source | **Done** |
| IDEAL-17.1 | Prompt registry | 17 | P0 | `check_agents_no_inline_prompts.py` | **Done** |
| IDEAL-17.2 | Prompt registry | 17 | P1 | Prompt version diff API | **Done** |
| IDEAL-17.3 | Prompt registry | 17 | P1 | Golden case per prompt version (extends FAUDIT-PE.1) | **Done** |
| IDEAL-17.4 | Prompt registry | 17 | P1 | Policy overlay composition test | **Done** |
| IDEAL-17.5 | Prompt registry | 17 | P1 | Prompt linkage in registry snapshot | **Done** |
| IDEAL-18.1 | Agent assembly | 18 | P1 | `check_agent_skill_resolution.py` (existing) | **Done** |
| IDEAL-18.2 | Agent assembly | 18 | P2 | `ModalityProfile` on contracts | **Done** |
| IDEAL-18.3 | Agent assembly | 18 | P1 | Cross-host reuse test | **Done** |
| IDEAL-18.4 | Agent assembly | 18 | P1 | Bounded loop max steps enforce | **Done** |
| IDEAL-19.1 | Registry | 19 | P1 | Lifecycle state on all artifact types | **Done** |
| IDEAL-19.2 | Registry | 19 | P1 | SemVer compatibility on resolve | **Done** |
| IDEAL-19.3 | Registry | 19 | P1 | Registry snapshot diff CI | **Done** |
| IDEAL-19.4 | Registry | 19 | P1 | Eval registry in assembly (FAUDIT-REG.1 extend) | **Done** |
| IDEAL-20.1 | Capability graph | 20 | P0 | `phase_v_capability_graph_guard.py --enforce` in §6.1 (existing) | **Done** |
| IDEAL-20.2 | Capability graph | 20 | P1 | Typed edge catalog | **Done** |
| IDEAL-20.3 | Capability graph | 20 | P1 | Tool change impact CLI report | **Done** |
| IDEAL-20.4 | Capability graph | 20 | P2 | Policy change visualization | **Done** |
| IDEAL-21.1 | Observability | 21 | P0 | `harness_slos.py` + link HARNESS_ENVIRONMENT SLO catalog | **Done** |
| IDEAL-21.2 | Observability | 21 | P0 | Runbook index in HARNESS_ENVIRONMENT (existing ORCH-5.5) | **Done** |
| IDEAL-21.3 | Observability | 21 | P1 | Cost dashboard metrics contract | **Done** |
| IDEAL-21.4 | Observability | 21 | P1 | Mandatory emission audit all run types | **Done** |
| IDEAL-21.5 | Observability | 21 | P2 | Product dashboard (GOV-PROD.1) | Deferred §6.3 |
| IDEAL-21.6 | Observability | 21 | P1 | OTLP profile on all reference hosts | **Done** |
| IDEAL-22.1 | Reliability | 22 | P0 | `harness_error_taxonomy.py` + expanded classifier | **Done** |
| IDEAL-22.2 | Reliability | 22 | P0 | Quality vs dependency recovery paths | **Done** |
| IDEAL-22.3 | Reliability | 22 | P1 | Compensation flow pattern | **Done** |
| IDEAL-22.4 | Reliability | 22 | P1 | Partial results contract | **Done** |
| IDEAL-22.5 | Reliability | 22 | P1 | Chaos/simulation test job | **Done** |
| IDEAL-22.6 | Reliability | 22 | P1 | Per-step retry budget | **Done** |
| IDEAL-23.1 | Security | 23 | P0 | `data_classification_enforcement.py` | **Done** |
| IDEAL-23.2 | Security | 23 | P0 | Prompt injection adversarial gate | **Done** |
| IDEAL-23.3 | Security | 23 | P1 | Tool injection defense gate | **Done** |
| IDEAL-23.4 | Security | 23 | P1 | Immutable security audit trail | **Done** |
| IDEAL-23.5 | Security | 23 | P1 | Retention per classification | **Done** |
| IDEAL-23.6 | Security | 23 | P1 | Output PII redaction middleware | **Done** |
| IDEAL-24.1 | Cost | 24 | P0 | `production_budget_policy.py` - mandatory `run_budget` | **Done** |
| IDEAL-24.2 | Cost | 24 | P1 | Per-tenant cost metrics | **Done** |
| IDEAL-24.3 | Cost | 24 | P2 | Token anomaly via W-ADAPT | **Done** |
| IDEAL-24.4 | Cost | 24 | P2 | Cost-aware model routing | **Done** |
| IDEAL-24.5 | Cost | 24 | P1 | Quota hard-stop vs warn | **Done** |
| IDEAL-25.1 | Evaluation | 25 | P1 | Golden scenario library | **Done** |
| IDEAL-25.2 | Evaluation | 25 | P1 | Version comparison CI artifacts | **Done** |
| IDEAL-25.3 | Evaluation | 25 | P2 | Shadow eval path | **Done** |
| IDEAL-25.4 | Evaluation | 25 | P2 | Human review sample queue | **Done** |
| IDEAL-25.5 | Evaluation | 25 | P1 | Context/RAG eval in release gate | **Done** |
| IDEAL-26.1 | CI / gates | 26 | P1 | `check_ideal_harness_l3_gates.py` umbrella | **Done** |
| IDEAL-26.2 | CI / gates | 26 | P2 | Weekly chaos job | **Done** |
| IDEAL-27.1 | DX | 27 | P1 | Trace Explorer filters | **Done** |
| IDEAL-27.2 | DX | 27 | P2 | Lab replay one-click | **Done** |
| IDEAL-27.3 | DX | 27 | P2 | `intergrax doctor` maturity score | **Done** |
| IDEAL-28.1 | Tier-3 hosts | 28 | P1 | Default task control on product hosts (H-APP-WIRING) | **Done** |
| IDEAL-28.2 | Tier-3 hosts | 28 | P1 | Durable async queue scaffold default | **Done** |
| IDEAL-28.3 | Tier-3 hosts | 28 | P2 | MVP promotion Tier-3 router | **Done** |
| IDEAL-28.4 | Tier-3 hosts | 28 | P4 | CFG-14 LKW daemon | Deferred §6.3 |
| IDEAL-29.1 | Modality | 29 | P2 | Vision remote host E2E | **Done** |
| IDEAL-29.2 | Modality | 29 | P1 | Media byte cap enforcement test | **Done** |
| IDEAL-29.3 | Modality | 29 | P1 | Vendor SDK import lint in agents | **Done** |
| IDEAL-30.1 | Ops / SLO | 30 | P1 | SLO breach incident classification | **Done** |
| IDEAL-30.2 | Ops / SLO | 30 | P1 | Four playbook runbooks (ideal §11.3) | **Done** |
| IDEAL-30.3 | Ops / SLO | 30 | P2 | PRR checklist in CI | **Done** |
| IDEAL-30.4 | Ops / SLO | 30 | P1 | `W_OPS_RELEASE_CYCLES>=2` deploy enforce | **Done** |
| IDEAL-31.1 | Agent lifecycle | 31 | P0 | Retired/deprecated routing block (existing) | **Done** |
| IDEAL-31.2 | Agent lifecycle | 31 | P1 | Certification checklist automation | **Done** |
| IDEAL-31.3 | Agent lifecycle | 31 | P1 | Promotion API experimental→certified | **Done** |
| IDEAL-31.4 | Agent lifecycle | 31 | P1 | Owner/on-call mandatory certified | **Done** |
| IDEAL-31.5 | Agent lifecycle | 31 | P2 | Deprecation sunset window | **Done** |
| IDEAL-32.1 | Doc governance | 32 | P1 | PR template layer checklist | **Done** |
| IDEAL-32.2 | Doc governance | 32 | P2 | Living architecture debt register | **Done** |
| IDEAL-32.3 | Doc governance | 32 | P1 | Maturity scorecard sync gate | **Done** |

---

## Domain routing

| Layers | Domain plan |
|--------|-------------|
| §4, §23–§24 | [`plan/UNIFIED_EXECUTION_RUNTIME.md`](UNIFIED_EXECUTION_RUNTIME.md) |
| §3, §9–§10 | [`plan/ORCHESTRATION.md`](ORCHESTRATION.md) |
| §21–§22, §30 | [`plan/OBSERVABILITY.md`](OBSERVABILITY.md) · [`plan/RELIABILITY_FAILURE_AND_HITL.md`](RELIABILITY_FAILURE_AND_HITL.md) |
| §14–§16 | [`plan/MEMORY.md`](MEMORY.md) |
| §17–§20, §31 | [`plan/AGENT_CONTRACTS_AND_ASSEMBLY.md`](AGENT_CONTRACTS_AND_ASSEMBLY.md) |
| §25–§27 | [`plan/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md`](EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md) · [`plan/CRITIC_VERIFICATION.md`](CRITIC_VERIFICATION.md) |
| §28 | [`plan/TIER3_APPLICATION_ENVIRONMENT.md`](TIER3_APPLICATION_ENVIRONMENT.md) |
| §6 | [`plan/LLM_ADAPTERS.md`](LLM_ADAPTERS.md) |
| §29 | [`plan/MODALITY.md`](MODALITY.md) |

---

## Completion criteria (phase IDEAL-L3)

Phase **IDEAL-L3** closes when:

1. All **P0** rows **Done** with gate evidence.
2. Critical ideal areas (Policy, Reliability, Observability ops) ≥ **L3** on Appendix M scorecard.
3. `uv run pytest -m gate -q` green.
4. `uv run python scripts/gates/check_ideal_harness_l3_gates.py` green.
5. P1 rows tracked as incremental maintenance (not blocking phase close).

**W2 close evidence (2026-06-09):** scorecard **32/32 L3** via `harness_maturity_report.py`; W2 gates in `test_ideal_harness_l3_w2_depth_gate.py` + umbrella script extensions.

**Quarterly review (IDEAL-1.2):** operator reviews `docs/project/maintainers/plans/IDEAL_HARNESS_L3.md` + `docs/project/technical/guides/ARCHITECTURE_DEBT_REGISTER.md` each quarter; update scorecard rows when layer maturity shifts.

**ADR policy:** No new ADR unless contract change - depth gates only; record **no ADR needed** on W2 close.

---

## Successor phase - AUDIT-IDEAL (post-L3 ideal gaps)

**Status:** **Planned** (2026-06-09) - baseline **32/32 L3** achieved; next uplift toward full [`IDEAL_HARNESS_AI_ARCHITECTURE.md`](../../technical/guides/IDEAL_HARNESS_AI_ARCHITECTURE.md) vision.

**Master register:** [`plan/AUDIT_IDEAL_2026.md`](AUDIT_IDEAL_2026.md) · Band **2az** · queue **§6.1au** in [`plan/PLATFORM_FOUNDATION.md`](PLATFORM_FOUNDATION.md)

**Waves:** W0 (register sync) → W1 P0 (org memory, ECP doc sync, registry durable) → W2 P1 (AHI evidence, reasoning, shadow eval, DX HTTP) → W3 P2 → W4 Band 3 product (§6.3 deferred).
