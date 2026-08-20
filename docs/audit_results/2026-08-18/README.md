# Audit campaign — 2026-08-18

**Protocol:** [Protocol v2](../AUDIT_PROTOCOL.md)  
**Canonical location:** `docs/audit_results`  
**Remediation rule:** audit first; remediation later after cross-layer review unless explicitly authorized.

## Campaign metadata

| Field | Value |
|-------|-------|
| `campaign_id` | `2026-08-18` |
| `campaign_token` | `20260818` |
| `started_at` | 2026-08-18 (UTC) |
| `completed_at` | — |
| `status` | `IN_PROGRESS` |
| `campaign_start_sha` | `9658224495c775fcefd55ab52bbcc7a94c84fb50` |
| `campaign_end_sha` | — |
| `scope` | Platform audit — fifteen completed audit units (`STRATEGIC_HARNESS_MODEL`, `TIER_LAYER_BOUNDARIES`, `PROVIDER_BACKEND_ABSTRACTION`, `INTERFACE_TASK_INTAKE`, `IDENTITY_TRUST`, `POLICY_GOVERNANCE`, `LLM_ADAPTERS`, `REASONING_PLANNING`, `EXECUTION_RUNTIME`, `PLATFORM_FOUNDATION`, `ORCHESTRATION`, `AGENT_SYSTEM`, `TOOLS`, `SKILLS`, `CODE_CRAFT`) |
| `overall_verdict` | — |
| `audit_method` | falsification-first, evidence-driven, no preference for PASS or FAIL |
| `operator_decision` | STRATEGIC_HARNESS_MODEL accepted 2026-08-18; TIER_LAYER_BOUNDARIES accepted 2026-08-18; PROVIDER_BACKEND_ABSTRACTION accepted 2026-08-18; INTERFACE_TASK_INTAKE accepted 2026-08-18; IDENTITY_TRUST accepted 2026-08-18; POLICY_GOVERNANCE accepted 2026-08-19; LLM_ADAPTERS accepted 2026-08-19; REASONING_PLANNING accepted 2026-08-19; EXECUTION_RUNTIME accepted 2026-08-19; PLATFORM_FOUNDATION accepted 2026-08-19; ORCHESTRATION accepted 2026-08-20; AGENT_SYSTEM accepted 2026-08-20; TOOLS accepted 2026-08-20; SKILLS accepted 2026-08-20; CODE_CRAFT accepted 2026-08-20 |

Exact audit-start time was not captured before first Protocol v2 persistence; date-level UTC precision is preserved rather than fabricating a clock time.

`post_sync_sha` on each layer row identifies the commit that synchronized the canonical audit result with current target architecture and implementation plans. The whole campaign does **not** use one immutable SHA — each layer has its own `audited_sha`; `development` may advance between layers.

## Layer register

| layer | status | audited_sha | verdict | critical | high | medium | low | architecture_sync | plan_sync | post_sync_sha | report |
|-------|--------|-------------|---------|----------|------|--------|-----|---------------------|-----------|---------------|--------|
| STRATEGIC_HARNESS_MODEL | COMPLETE | `9658224495c775fcefd55ab52bbcc7a94c84fb50` | FAIL | 0 | 6 | 4 | 0 | COMPLETE | COMPLETE | `def29be1adf2e099c300b7a8471c32b946e9c957` | [STRATEGIC_HARNESS_MODEL.md](STRATEGIC_HARNESS_MODEL.md) |
| TIER_LAYER_BOUNDARIES | COMPLETE | `d8d10bb5099d003eb9495674c28e0f6e6762dbfa` | FAIL | 0 | 2 | 3 | 0 | COMPLETE | COMPLETE | `a5d6f83d0ea274dec269377a9ce1cc4421b1bd12` | [TIER_LAYER_BOUNDARIES.md](TIER_LAYER_BOUNDARIES.md) |
| PROVIDER_BACKEND_ABSTRACTION | COMPLETE | `7570e9b4508554a42bdf5cce2c987c56c6f2b80e` | FAIL | 0 | 2 | 3 | 0 | COMPLETE | COMPLETE | `3fb36254bf58f3898dac16f0ae0fca3f01bb95d6` | [PROVIDER_BACKEND_ABSTRACTION.md](PROVIDER_BACKEND_ABSTRACTION.md) |
| INTERFACE_TASK_INTAKE | COMPLETE | `2640d826da6f1a781e798326ff1b21b3a9f7c4cc` | FAIL | 0 | 3 | 3 | 0 | COMPLETE | COMPLETE | `f2550615df385e474508e08ce763b43cef7e980b` | [INTERFACE_TASK_INTAKE.md](INTERFACE_TASK_INTAKE.md) |
| IDENTITY_TRUST | COMPLETE | `6fbc5e4928963ecd386456158b0753662fed209b` | FAIL | 0 | 4 | 2 | 0 | COMPLETE | COMPLETE | `be52ca045443e906ef03f47fbd8cde1dbd1f6fbc` | [IDENTITY_TRUST.md](IDENTITY_TRUST.md) |
| POLICY_GOVERNANCE | COMPLETE | `042cc9b50386cfcd4da30310c84d000dbf5d2718` | FAIL | 0 | 4 | 1 | 0 | COMPLETE | COMPLETE | `d7988045cfa550c4338eedc326b54933c4058541` | [POLICY_GOVERNANCE.md](POLICY_GOVERNANCE.md) |
| LLM_ADAPTERS | COMPLETE | `b1e4de1d776acc64e8461f7dcdce09cd03d07b80` | FAIL | 0 | 4 | 2 | 0 | COMPLETE | COMPLETE | `d7988045cfa550c4338eedc326b54933c4058541` | [LLM_ADAPTERS.md](LLM_ADAPTERS.md) |
| REASONING_PLANNING | COMPLETE | `fe876d301df07ce22e438b0a55167275ccec32b5` | FAIL | 0 | 4 | 2 | 0 | COMPLETE | COMPLETE | `d7988045cfa550c4338eedc326b54933c4058541` | [REASONING_PLANNING.md](REASONING_PLANNING.md) |
| EXECUTION_RUNTIME | COMPLETE | `df7aaac19b20e84c06d6233492cdb4365a892f4f` | FAIL | 0 | 5 | 1 | 0 | COMPLETE | COMPLETE | `d7988045cfa550c4338eedc326b54933c4058541` | [EXECUTION_RUNTIME.md](EXECUTION_RUNTIME.md) |
| PLATFORM_FOUNDATION | COMPLETE | `f21d5c3dc417907acb50d597642d3892e704bd47` | FAIL | 0 | 5 | 0 | 1 | COMPLETE | COMPLETE | `60eff55ca7105cc8d277201c95785b4c037e3bd9` | [PLATFORM_FOUNDATION.md](PLATFORM_FOUNDATION.md) |
| ORCHESTRATION | COMPLETE | `a784966681782bc58412af290c2978c1d1f152a3` | FAIL | 0 | 4 | 1 | 0 | COMPLETE | COMPLETE | `b33d5a5b5755ed76038ee10b51b90efbda34ecd7` | [ORCHESTRATION.md](ORCHESTRATION.md) |
| AGENT_SYSTEM | COMPLETE | `654a7c0e3fe823a43a2620645848248023e1c64e` | FAIL | 0 | 5 | 1 | 0 | COMPLETE | COMPLETE | `995c34ee5b9ca355e9bf3ec02c425f51d6cedaf4` | [AGENT_SYSTEM.md](AGENT_SYSTEM.md) |
| TOOLS | COMPLETE | `65aaf33a6a6dba9b336162ec547cd677f4edad91` | FAIL | 0 | 5 | 1 | 0 | COMPLETE | COMPLETE | `c6cc72056c60730516e6228856dbd1c8611c8c46` | [TOOLS.md](TOOLS.md) |
| SKILLS | COMPLETE | `2df2f07d10aa19c4d62694f21858be501a3d6d18` | FAIL | 0 | 3 | 3 | 0 | COMPLETE | COMPLETE | `1d17272ceb2f486320e7265bfd62ca872961d74b` | [SKILLS.md](SKILLS.md) |
| CODE_CRAFT | COMPLETE | `f985ad342d0d6db38c9998df67f9cd7bc10bfa46` | FAIL | 2 | 5 | 0 | 0 | COMPLETE | COMPLETE | — | [CODE_CRAFT.md](CODE_CRAFT.md) |

## Finding register

Authoritative current lifecycle for remediation. Immutable observation and evidence remain in per-layer reports.

| finding_id | layer | severity | category | status | remediation_block | dependencies | arch_ref | plan_ref | implementation_commit | verification_evidence | notes |
|------------|-------|----------|----------|--------|-------------------|--------------|----------|----------|----------------------|-------------------------|-------|
| AUDIT-20260818-STRATEGIC_HARNESS_MODEL-01 | STRATEGIC_HARNESS_MODEL | HIGH | ARCHITECTURE DEFECT | ACCEPTED | SHM-FIX-A | — | `docs/project/architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md` — [Protocol v2 strategic harness target invariants (2026-08-18)](#protocol-v2-strategic-harness-target-invariants-2026-08-18); `docs/project/architecture/UNIFIED_EXECUTION_RUNTIME.md` — [Protocol v2 strategic harness target invariants (2026-08-18)](#protocol-v2-strategic-harness-target-invariants-2026-08-18) | `docs/project/maintainers/plans/AGENT_CONTRACTS_AND_ASSEMBLY.md` — SHM-FIX-A; `docs/project/maintainers/plans/UNIFIED_EXECUTION_RUNTIME.md` — SHM-FIX-A | — | — | operator accepted 2026-08-18 |
| AUDIT-20260818-STRATEGIC_HARNESS_MODEL-02 | STRATEGIC_HARNESS_MODEL | HIGH | BOUNDARY VIOLATION | ACCEPTED | SHM-FIX-A | — | `docs/project/architecture/UNIFIED_EXECUTION_RUNTIME.md` — [Protocol v2 strategic harness target invariants (2026-08-18)](#protocol-v2-strategic-harness-target-invariants-2026-08-18) | `docs/project/maintainers/plans/UNIFIED_EXECUTION_RUNTIME.md` — SHM-FIX-A | — | — | operator accepted 2026-08-18 |
| AUDIT-20260818-STRATEGIC_HARNESS_MODEL-03 | STRATEGIC_HARNESS_MODEL | MEDIUM | BOUNDARY VIOLATION | ACCEPTED | SHM-FIX-A | — | `docs/project/architecture/UNIFIED_EXECUTION_RUNTIME.md` — [Protocol v2 strategic harness target invariants (2026-08-18)](#protocol-v2-strategic-harness-target-invariants-2026-08-18) | `docs/project/maintainers/plans/UNIFIED_EXECUTION_RUNTIME.md` — SHM-FIX-A | — | — | operator accepted 2026-08-18 |
| AUDIT-20260818-STRATEGIC_HARNESS_MODEL-04 | STRATEGIC_HARNESS_MODEL | HIGH | IMPLEMENTATION/ARCHITECTURE DRIFT | ACCEPTED | SHM-FIX-A | — | `docs/project/architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md` — [Protocol v2 strategic harness target invariants (2026-08-18)](#protocol-v2-strategic-harness-target-invariants-2026-08-18) | `docs/project/maintainers/plans/AGENT_CONTRACTS_AND_ASSEMBLY.md` — SHM-FIX-A | — | — | operator accepted 2026-08-18 |
| AUDIT-20260818-STRATEGIC_HARNESS_MODEL-05 | STRATEGIC_HARNESS_MODEL | MEDIUM | ARCHITECTURE DEFECT | ACCEPTED | SHM-FIX-C | — | `docs/project/architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md` — [Protocol v2 strategic harness target invariants (2026-08-18)](#protocol-v2-strategic-harness-target-invariants-2026-08-18); `docs/project/architecture/UNIFIED_EXECUTION_RUNTIME.md` — [Protocol v2 strategic harness target invariants (2026-08-18)](#protocol-v2-strategic-harness-target-invariants-2026-08-18) | `docs/project/maintainers/plans/AGENT_CONTRACTS_AND_ASSEMBLY.md` — SHM-FIX-C; `docs/project/maintainers/plans/UNIFIED_EXECUTION_RUNTIME.md` — SHM-FIX-C | — | — | operator accepted 2026-08-18 |
| AUDIT-20260818-STRATEGIC_HARNESS_MODEL-06 | STRATEGIC_HARNESS_MODEL | HIGH | IMPLEMENTATION/ARCHITECTURE DRIFT | ACCEPTED | SHM-FIX-B | — | `docs/project/architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md` — [Protocol v2 strategic harness target invariants (2026-08-18)](#protocol-v2-strategic-harness-target-invariants-2026-08-18) | `docs/project/maintainers/plans/AGENT_CONTRACTS_AND_ASSEMBLY.md` — SHM-FIX-B | — | — | operator accepted 2026-08-18 |
| AUDIT-20260818-STRATEGIC_HARNESS_MODEL-07 | STRATEGIC_HARNESS_MODEL | MEDIUM | BOUNDARY VIOLATION | ACCEPTED | SHM-FIX-C | — | `docs/project/architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md` — [Protocol v2 strategic harness target invariants (2026-08-18)](#protocol-v2-strategic-harness-target-invariants-2026-08-18) | `docs/project/maintainers/plans/AGENT_CONTRACTS_AND_ASSEMBLY.md` — SHM-FIX-C | — | — | operator accepted 2026-08-18 |
| AUDIT-20260818-STRATEGIC_HARNESS_MODEL-08 | STRATEGIC_HARNESS_MODEL | HIGH | RELIABILITY | ACCEPTED | SHM-FIX-B | — | `docs/project/architecture/UNIFIED_EXECUTION_RUNTIME.md` — [Protocol v2 strategic harness target invariants (2026-08-18)](#protocol-v2-strategic-harness-target-invariants-2026-08-18) | `docs/project/maintainers/plans/UNIFIED_EXECUTION_RUNTIME.md` — SHM-FIX-B | — | — | operator accepted 2026-08-18 |
| AUDIT-20260818-STRATEGIC_HARNESS_MODEL-09 | STRATEGIC_HARNESS_MODEL | HIGH | IMPLEMENTATION DEFECT | ACCEPTED | SHM-FIX-B | — | `docs/project/architecture/UNIFIED_EXECUTION_RUNTIME.md` — [Protocol v2 strategic harness target invariants (2026-08-18)](#protocol-v2-strategic-harness-target-invariants-2026-08-18) | `docs/project/maintainers/plans/UNIFIED_EXECUTION_RUNTIME.md` — SHM-FIX-B | — | — | Related classification: TEST GAP; operator accepted 2026-08-18 |
| AUDIT-20260818-STRATEGIC_HARNESS_MODEL-10 | STRATEGIC_HARNESS_MODEL | MEDIUM | PROCESS / CLAIM | ACCEPTED | SHM-FIX-D | SHM-FIX-A, SHM-FIX-B, SHM-FIX-C | `docs/project/architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md` — [Protocol v2 strategic harness target invariants (2026-08-18)](#protocol-v2-strategic-harness-target-invariants-2026-08-18); `docs/project/architecture/UNIFIED_EXECUTION_RUNTIME.md` — [Protocol v2 strategic harness target invariants (2026-08-18)](#protocol-v2-strategic-harness-target-invariants-2026-08-18) | `docs/project/maintainers/plans/AGENT_CONTRACTS_AND_ASSEMBLY.md` — SHM-FIX-D; `docs/project/maintainers/plans/UNIFIED_EXECUTION_RUNTIME.md` — SHM-FIX-D | — | — | operator accepted 2026-08-18 |
| AUDIT-20260818-TIER_LAYER_BOUNDARIES-01 | TIER_LAYER_BOUNDARIES | HIGH | IMPLEMENTATION DEFECT | ACCEPTED | TL-FIX-A | — | `docs/project/architecture/PLATFORM_FOUNDATION.md` — [Protocol v2 tier-boundary target invariants (2026-08-18)](#protocol-v2-tier-boundary-target-invariants-2026-08-18) | `docs/project/maintainers/plans/PLATFORM_FOUNDATION.md` — TL-FIX-A | — | — | operator accepted 2026-08-18 |
| AUDIT-20260818-TIER_LAYER_BOUNDARIES-02 | TIER_LAYER_BOUNDARIES | HIGH | BOUNDARY VIOLATION | ACCEPTED | TL-FIX-B | — | `docs/project/architecture/AGENT_DISTRIBUTION.md` — [Protocol v2 agent ownership target invariants (2026-08-18)](#protocol-v2-agent-ownership-target-invariants-2026-08-18) | `docs/project/maintainers/plans/AGENT_DISTRIBUTION.md` — TL-FIX-B | — | — | operator accepted 2026-08-18 |
| AUDIT-20260818-TIER_LAYER_BOUNDARIES-03 | TIER_LAYER_BOUNDARIES | MEDIUM | BOUNDARY VIOLATION | ACCEPTED | TL-FIX-C | — | `docs/project/architecture/TIER3_APPLICATION_ENVIRONMENT.md` — [Protocol v2 Tier-3 boundary target invariants (2026-08-18)](#protocol-v2-tier3-boundary-target-invariants-2026-08-18) | `docs/project/maintainers/plans/TIER3_APPLICATION_ENVIRONMENT.md` — TL-FIX-C | — | — | operator accepted 2026-08-18 |
| AUDIT-20260818-TIER_LAYER_BOUNDARIES-04 | TIER_LAYER_BOUNDARIES | MEDIUM | BOUNDARY VIOLATION | ACCEPTED | TL-FIX-D | — | `docs/project/architecture/TIER3_APPLICATION_ENVIRONMENT.md` — [Protocol v2 Tier-3 boundary target invariants (2026-08-18)](#protocol-v2-tier3-boundary-target-invariants-2026-08-18) | `docs/project/maintainers/plans/TIER3_APPLICATION_ENVIRONMENT.md` — TL-FIX-D | — | — | operator accepted 2026-08-18 |
| AUDIT-20260818-TIER_LAYER_BOUNDARIES-05 | TIER_LAYER_BOUNDARIES | MEDIUM | TEST GAP | ACCEPTED | TL-FIX-A | — | `docs/project/architecture/PLATFORM_FOUNDATION.md` — [Protocol v2 tier-boundary target invariants (2026-08-18)](#protocol-v2-tier-boundary-target-invariants-2026-08-18); `docs/project/architecture/TIER3_APPLICATION_ENVIRONMENT.md` — [Protocol v2 Tier-3 boundary target invariants (2026-08-18)](#protocol-v2-tier3-boundary-target-invariants-2026-08-18) | `docs/project/maintainers/plans/PLATFORM_FOUNDATION.md` — TL-FIX-A | — | — | operator accepted 2026-08-18 |
| AUDIT-20260818-PROVIDER_BACKEND_ABSTRACTION-01 | PROVIDER_BACKEND_ABSTRACTION | HIGH | BOUNDARY VIOLATION | ACCEPTED | PBA-FIX-A | — | `docs/project/architecture/RELIABILITY_FAILURE_AND_HITL.md` — [Protocol v2.2 provider/backend abstraction target invariants (2026-08-18)](#protocol-v22-provider-backend-abstraction-target-invariants-2026-08-18) | `docs/project/maintainers/plans/RELIABILITY_FAILURE_AND_HITL.md` — PBA-FIX-A | — | — | operator accepted 2026-08-18; PAPER_ABSTRACTION / VENDOR_LEAK |
| AUDIT-20260818-PROVIDER_BACKEND_ABSTRACTION-02 | PROVIDER_BACKEND_ABSTRACTION | HIGH | BOUNDARY VIOLATION | ACCEPTED | PBA-FIX-B | — | `docs/project/architecture/INTEGRATIONS.md` — [Protocol v2.2 provider/backend abstraction target invariants (2026-08-18)](#protocol-v22-provider-backend-abstraction-target-invariants-2026-08-18) | `docs/project/maintainers/plans/INTEGRATIONS.md` — PBA-FIX-B | — | — | operator accepted 2026-08-18; VENDOR_LEAK / BYPASSED PLATFORM MECHANISM |
| AUDIT-20260818-PROVIDER_BACKEND_ABSTRACTION-03 | PROVIDER_BACKEND_ABSTRACTION | MEDIUM | ARCHITECTURE DEFECT | ACCEPTED | PBA-FIX-C | — | `docs/project/architecture/INTEGRATIONS.md` — [Protocol v2.2 provider/backend abstraction target invariants (2026-08-18)](#protocol-v22-provider-backend-abstraction-target-invariants-2026-08-18) | `docs/project/maintainers/plans/INTEGRATIONS.md` — PBA-FIX-C | — | — | operator accepted 2026-08-18; VENDOR_LEAK |
| AUDIT-20260818-PROVIDER_BACKEND_ABSTRACTION-04 | PROVIDER_BACKEND_ABSTRACTION | MEDIUM | TEST GAP | ACCEPTED | PBA-FIX-B | — | `docs/project/architecture/INTEGRATIONS.md` — [Protocol v2.2 provider/backend abstraction target invariants (2026-08-18)](#protocol-v22-provider-backend-abstraction-target-invariants-2026-08-18) | `docs/project/maintainers/plans/INTEGRATIONS.md` — PBA-FIX-B | — | — | operator accepted 2026-08-18 |
| AUDIT-20260818-PROVIDER_BACKEND_ABSTRACTION-05 | PROVIDER_BACKEND_ABSTRACTION | MEDIUM | ARCHITECTURE DEFECT | ACCEPTED | PBA-FIX-D | — | `docs/project/architecture/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md` — [Protocol v2.2 provider/backend abstraction target invariants (2026-08-18)](#protocol-v22-provider-backend-abstraction-target-invariants-2026-08-18) | `docs/project/maintainers/plans/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md` — PBA-FIX-D | — | — | operator accepted 2026-08-18; MISSING_ABSTRACTION / VENDOR_LEAK |
| AUDIT-20260818-INTERFACE_TASK_INTAKE-01 | INTERFACE_TASK_INTAKE | HIGH | IMPLEMENTATION/ARCHITECTURE DRIFT | ACCEPTED | ITI-FIX-A | — | `docs/project/architecture/TIER3_APPLICATION_ENVIRONMENT.md` — [Protocol v2.2 Tier-3 intake target invariants (2026-08-18)](#protocol-v22-tier3-intake-target-invariants-2026-08-18) | `docs/project/maintainers/plans/TIER3_APPLICATION_ENVIRONMENT.md` — ITI-FIX-A | — | — | operator accepted 2026-08-18 |
| AUDIT-20260818-INTERFACE_TASK_INTAKE-02 | INTERFACE_TASK_INTAKE | HIGH | IMPLEMENTATION DEFECT | ACCEPTED | ITI-FIX-B | — | `docs/project/architecture/NEXUS_EXECUTION_FLOW.md` — [Protocol v2.2 task-intake execution convergence target invariants (2026-08-18)](#protocol-v22-task-intake-execution-convergence-target-invariants-2026-08-18) | `docs/project/maintainers/plans/NEXUS_EXECUTION_FLOW.md` — ITI-FIX-B | — | — | Related classification: TEST GAP; operator accepted 2026-08-18 |
| AUDIT-20260818-INTERFACE_TASK_INTAKE-03 | INTERFACE_TASK_INTAKE | HIGH | BOUNDARY VIOLATION | ACCEPTED | ITI-FIX-C | — | `docs/project/architecture/NEXUS_EXECUTION_FLOW.md` — [Protocol v2.2 task-intake execution convergence target invariants (2026-08-18)](#protocol-v22-task-intake-execution-convergence-target-invariants-2026-08-18); `docs/project/architecture/TIER3_APPLICATION_ENVIRONMENT.md` — [Protocol v2.2 Tier-3 intake target invariants (2026-08-18)](#protocol-v22-tier3-intake-target-invariants-2026-08-18) | `docs/project/maintainers/plans/NEXUS_EXECUTION_FLOW.md` — ITI-FIX-C | — | — | operator accepted 2026-08-18 |
| AUDIT-20260818-INTERFACE_TASK_INTAKE-04 | INTERFACE_TASK_INTAKE | MEDIUM | IMPLEMENTATION/ARCHITECTURE DRIFT | ACCEPTED | ITI-FIX-A | — | `docs/project/architecture/TIER3_APPLICATION_ENVIRONMENT.md` — [Protocol v2.2 Tier-3 intake target invariants (2026-08-18)](#protocol-v22-tier3-intake-target-invariants-2026-08-18) | `docs/project/maintainers/plans/TIER3_APPLICATION_ENVIRONMENT.md` — ITI-FIX-A | — | — | operator accepted 2026-08-18 |
| AUDIT-20260818-INTERFACE_TASK_INTAKE-05 | INTERFACE_TASK_INTAKE | MEDIUM | ARCHITECTURE DEFECT | ACCEPTED | ITI-FIX-C | — | `docs/project/architecture/NEXUS_EXECUTION_FLOW.md` — [Protocol v2.2 task-intake execution convergence target invariants (2026-08-18)](#protocol-v22-task-intake-execution-convergence-target-invariants-2026-08-18) | `docs/project/maintainers/plans/NEXUS_EXECUTION_FLOW.md` — ITI-FIX-C | — | — | operator accepted 2026-08-18 |
| AUDIT-20260818-INTERFACE_TASK_INTAKE-06 | INTERFACE_TASK_INTAKE | MEDIUM | TEST GAP | ACCEPTED | ITI-FIX-D | ITI-FIX-A, ITI-FIX-B, ITI-FIX-C | `docs/project/architecture/TIER3_APPLICATION_ENVIRONMENT.md` — [Protocol v2.2 Tier-3 intake target invariants (2026-08-18)](#protocol-v22-tier3-intake-target-invariants-2026-08-18) | `docs/project/maintainers/plans/TIER3_APPLICATION_ENVIRONMENT.md` — ITI-FIX-D | — | — | Related classification: PROCESS / CLAIM; operator accepted 2026-08-18 |
| AUDIT-20260818-IDENTITY_TRUST-01 | IDENTITY_TRUST | HIGH | SECURITY | ACCEPTED | IDT-FIX-A | — | `docs/project/architecture/TIER3_APPLICATION_ENVIRONMENT.md` — [Protocol v2.2 identity/trust target invariants (2026-08-18)](#protocol-v22-identitytrust-target-invariants-2026-08-18); `docs/project/architecture/UNIFIED_EXECUTION_RUNTIME.md` — [Protocol v2.2 execution identity closure target invariants (2026-08-18)](#protocol-v22-execution-identity-closure-target-invariants-2026-08-18) | `docs/project/maintainers/plans/TIER3_APPLICATION_ENVIRONMENT.md` — IDT-FIX-A | — | — | Related classification: TEST GAP; operator accepted 2026-08-18 |
| AUDIT-20260818-IDENTITY_TRUST-02 | IDENTITY_TRUST | HIGH | SECURITY | ACCEPTED | IDT-FIX-B | — | `docs/project/architecture/NEXUS_EXECUTION_FLOW.md` — [Protocol v2.2 delegated authority target invariants (2026-08-18)](#protocol-v22-delegated-authority-target-invariants-2026-08-18) | `docs/project/maintainers/plans/NEXUS_EXECUTION_FLOW.md` — IDT-FIX-B | — | — | operator accepted 2026-08-18 |
| AUDIT-20260818-IDENTITY_TRUST-03 | IDENTITY_TRUST | HIGH | SECURITY | ACCEPTED | IDT-FIX-C | — | `docs/project/architecture/RELIABILITY_FAILURE_AND_HITL.md` — [Protocol v2.2 human decision provenance target invariants (2026-08-18)](#protocol-v22-human-decision-provenance-target-invariants-2026-08-18) | `docs/project/maintainers/plans/RELIABILITY_FAILURE_AND_HITL.md` — IDT-FIX-C | — | — | operator accepted 2026-08-18 |
| AUDIT-20260818-IDENTITY_TRUST-04 | IDENTITY_TRUST | MEDIUM | IMPLEMENTATION DEFECT | ACCEPTED | IDT-FIX-C | — | `docs/project/architecture/RELIABILITY_FAILURE_AND_HITL.md` — [Protocol v2.2 human decision provenance target invariants (2026-08-18)](#protocol-v22-human-decision-provenance-target-invariants-2026-08-18) | `docs/project/maintainers/plans/RELIABILITY_FAILURE_AND_HITL.md` — IDT-FIX-C | — | — | operator accepted 2026-08-18 |
| AUDIT-20260818-IDENTITY_TRUST-05 | IDENTITY_TRUST | HIGH | IMPLEMENTATION DEFECT | ACCEPTED | IDT-FIX-D | — | `docs/project/architecture/UNIFIED_EXECUTION_RUNTIME.md` — [Protocol v2.2 execution identity closure target invariants (2026-08-18)](#protocol-v22-execution-identity-closure-target-invariants-2026-08-18) | `docs/project/maintainers/plans/UNIFIED_EXECUTION_RUNTIME.md` — IDT-FIX-D | — | — | Related classification: OPERABILITY; operator accepted 2026-08-18 |
| AUDIT-20260818-IDENTITY_TRUST-06 | IDENTITY_TRUST | MEDIUM | ARCHITECTURE DEFECT | ACCEPTED | IDT-FIX-A | — | `docs/project/architecture/TIER3_APPLICATION_ENVIRONMENT.md` — [Protocol v2.2 identity/trust target invariants (2026-08-18)](#protocol-v22-identitytrust-target-invariants-2026-08-18); `docs/project/architecture/UNIFIED_EXECUTION_RUNTIME.md` — [Protocol v2.2 execution identity closure target invariants (2026-08-18)](#protocol-v22-execution-identity-closure-target-invariants-2026-08-18) | `docs/project/maintainers/plans/TIER3_APPLICATION_ENVIRONMENT.md` — IDT-FIX-A | — | — | operator accepted 2026-08-18 |

| AUDIT-20260818-POLICY_GOVERNANCE-01 | POLICY_GOVERNANCE | HIGH | ARCHITECTURE DEFECT | ACCEPTED | PG-FIX-A | — | `docs/project/architecture/GOVERNED_EXECUTION.md` | `docs/project/maintainers/plans/GOVERNED_EXECUTION.md` — PG-FIX-A | — | — | operator accepted 2026-08-19 |
| AUDIT-20260818-POLICY_GOVERNANCE-02 | POLICY_GOVERNANCE | HIGH | SECURITY | ACCEPTED | PG-FIX-B | — | `docs/project/architecture/GOVERNED_EXECUTION.md` | `docs/project/maintainers/plans/GOVERNED_EXECUTION.md` — PG-FIX-B | — | — | operator accepted 2026-08-19 |
| AUDIT-20260818-POLICY_GOVERNANCE-03 | POLICY_GOVERNANCE | HIGH | ARCHITECTURE DEFECT | ACCEPTED | PG-FIX-A | — | `docs/project/architecture/GOVERNED_EXECUTION.md` | `docs/project/maintainers/plans/GOVERNED_EXECUTION.md` — PG-FIX-A | — | — | operator accepted 2026-08-19 |
| AUDIT-20260818-POLICY_GOVERNANCE-04 | POLICY_GOVERNANCE | HIGH | IMPLEMENTATION/ARCHITECTURE DRIFT | ACCEPTED | PG-FIX-C | — | `docs/project/architecture/GOVERNED_EXECUTION.md` | `docs/project/maintainers/plans/GOVERNED_EXECUTION.md` — PG-FIX-C | — | — | operator accepted 2026-08-19 |
| AUDIT-20260818-POLICY_GOVERNANCE-05 | POLICY_GOVERNANCE | MEDIUM | ARCHITECTURE DEFECT | ACCEPTED | PG-FIX-D | — | `docs/project/architecture/GOVERNED_EXECUTION.md` | `docs/project/maintainers/plans/GOVERNED_EXECUTION.md` — PG-FIX-D | — | — | operator accepted 2026-08-19 |
| AUDIT-20260818-LLM_ADAPTERS-01 | LLM_ADAPTERS | HIGH | BOUNDARY VIOLATION | ACCEPTED | LLM-FIX-A | — | `docs/project/architecture/NEXUS_EXECUTION_FLOW.md` | `docs/project/maintainers/plans/NEXUS_EXECUTION_FLOW.md` — LLM-FIX-A | — | — | operator accepted 2026-08-19 |
| AUDIT-20260818-LLM_ADAPTERS-02 | LLM_ADAPTERS | HIGH | IMPLEMENTATION/ARCHITECTURE DRIFT | ACCEPTED | LLM-FIX-B | — | `docs/project/architecture/TIER3_APPLICATION_ENVIRONMENT.md` | `docs/project/maintainers/plans/TIER3_APPLICATION_ENVIRONMENT.md` — LLM-FIX-B | — | — | operator accepted 2026-08-19 |
| AUDIT-20260818-LLM_ADAPTERS-03 | LLM_ADAPTERS | HIGH | SECURITY | ACCEPTED | LLM-FIX-C | — | `docs/project/architecture/TIER3_APPLICATION_ENVIRONMENT.md` | `docs/project/maintainers/plans/TIER3_APPLICATION_ENVIRONMENT.md` — LLM-FIX-C | — | — | operator accepted 2026-08-19 |
| AUDIT-20260818-LLM_ADAPTERS-04 | LLM_ADAPTERS | HIGH | BOUNDARY VIOLATION | ACCEPTED | LLM-FIX-A | — | `docs/project/architecture/NEXUS_EXECUTION_FLOW.md` | `docs/project/maintainers/plans/NEXUS_EXECUTION_FLOW.md` — LLM-FIX-A | — | — | operator accepted 2026-08-19 |
| AUDIT-20260818-LLM_ADAPTERS-05 | LLM_ADAPTERS | MEDIUM | OPERABILITY | ACCEPTED | LLM-FIX-B | — | `docs/project/architecture/TIER3_APPLICATION_ENVIRONMENT.md` | `docs/project/maintainers/plans/TIER3_APPLICATION_ENVIRONMENT.md` — LLM-FIX-B | — | — | operator accepted 2026-08-19 |
| AUDIT-20260818-LLM_ADAPTERS-06 | LLM_ADAPTERS | MEDIUM | IMPLEMENTATION DEFECT | ACCEPTED | LLM-FIX-D | — | `docs/project/architecture/NEXUS_EXECUTION_FLOW.md` | `docs/project/maintainers/plans/NEXUS_EXECUTION_FLOW.md` — LLM-FIX-D | — | — | operator accepted 2026-08-19 |
| AUDIT-20260818-REASONING_PLANNING-01 | REASONING_PLANNING | HIGH | RELIABILITY | ACCEPTED | RPL-FIX-A | — | `docs/project/architecture/NEXUS_EXECUTION_FLOW.md` | `docs/project/maintainers/plans/NEXUS_EXECUTION_FLOW.md` — RPL-FIX-A | — | — | operator accepted 2026-08-19 |
| AUDIT-20260818-REASONING_PLANNING-02 | REASONING_PLANNING | HIGH | BOUNDARY VIOLATION | ACCEPTED | RPL-FIX-B | — | `docs/project/architecture/NEXUS_EXECUTION_FLOW.md` | `docs/project/maintainers/plans/NEXUS_EXECUTION_FLOW.md` — RPL-FIX-B | — | — | operator accepted 2026-08-19 |
| AUDIT-20260818-REASONING_PLANNING-03 | REASONING_PLANNING | HIGH | IMPLEMENTATION DEFECT | ACCEPTED | RPL-FIX-C | — | `docs/project/architecture/REASONING_AND_COGNITION.md` | `docs/project/maintainers/plans/REASONING_AND_COGNITION.md` — RPL-FIX-C | — | — | operator accepted 2026-08-19 |
| AUDIT-20260818-REASONING_PLANNING-04 | REASONING_PLANNING | HIGH | IMPLEMENTATION DEFECT | ACCEPTED | RPL-FIX-D | — | `docs/project/architecture/REASONING_AND_COGNITION.md` | `docs/project/maintainers/plans/REASONING_AND_COGNITION.md` — RPL-FIX-D | — | — | operator accepted 2026-08-19 |
| AUDIT-20260818-REASONING_PLANNING-05 | REASONING_PLANNING | MEDIUM | RELIABILITY | ACCEPTED | RPL-FIX-E | — | `docs/project/architecture/REASONING_AND_COGNITION.md` | `docs/project/maintainers/plans/REASONING_AND_COGNITION.md` — RPL-FIX-E | — | — | operator accepted 2026-08-19 |
| AUDIT-20260818-REASONING_PLANNING-06 | REASONING_PLANNING | MEDIUM | BOUNDARY VIOLATION | ACCEPTED | RPL-FIX-F | — | `docs/project/architecture/REASONING_AND_COGNITION.md` | `docs/project/maintainers/plans/REASONING_AND_COGNITION.md` — RPL-FIX-F | — | — | operator accepted 2026-08-19 |
| AUDIT-20260818-EXECUTION_RUNTIME-01 | EXECUTION_RUNTIME | HIGH | BOUNDARY VIOLATION | ACCEPTED | UER-FIX-A | — | `docs/project/architecture/UNIFIED_EXECUTION_RUNTIME.md` | `docs/project/maintainers/plans/UNIFIED_EXECUTION_RUNTIME.md` — UER-FIX-A | — | — | operator accepted 2026-08-19 |
| AUDIT-20260818-EXECUTION_RUNTIME-02 | EXECUTION_RUNTIME | HIGH | RELIABILITY | ACCEPTED | UER-FIX-B | — | `docs/project/architecture/UNIFIED_EXECUTION_RUNTIME.md` | `docs/project/maintainers/plans/UNIFIED_EXECUTION_RUNTIME.md` — UER-FIX-B | — | — | operator accepted 2026-08-19 |
| AUDIT-20260818-EXECUTION_RUNTIME-03 | EXECUTION_RUNTIME | HIGH | IMPLEMENTATION/ARCHITECTURE DRIFT | ACCEPTED | UER-FIX-C | — | `docs/project/architecture/UNIFIED_EXECUTION_RUNTIME.md` | `docs/project/maintainers/plans/UNIFIED_EXECUTION_RUNTIME.md` — UER-FIX-C | — | — | operator accepted 2026-08-19 |
| AUDIT-20260818-EXECUTION_RUNTIME-04 | EXECUTION_RUNTIME | HIGH | RELIABILITY | ACCEPTED | UER-FIX-D | — | `docs/project/architecture/UNIFIED_EXECUTION_RUNTIME.md` | `docs/project/maintainers/plans/UNIFIED_EXECUTION_RUNTIME.md` — UER-FIX-D | — | — | operator accepted 2026-08-19 |
| AUDIT-20260818-EXECUTION_RUNTIME-05 | EXECUTION_RUNTIME | HIGH | RELIABILITY | ACCEPTED | UER-FIX-E | — | `docs/project/architecture/RELIABILITY_FAILURE_AND_HITL.md` | `docs/project/maintainers/plans/RELIABILITY_FAILURE_AND_HITL.md` — UER-FIX-E | — | — | operator accepted 2026-08-19 |
| AUDIT-20260818-EXECUTION_RUNTIME-06 | EXECUTION_RUNTIME | MEDIUM | RELIABILITY | ACCEPTED | UER-FIX-E | — | `docs/project/architecture/RELIABILITY_FAILURE_AND_HITL.md` | `docs/project/maintainers/plans/RELIABILITY_FAILURE_AND_HITL.md` — UER-FIX-E | — | — | operator accepted 2026-08-19 |
| AUDIT-20260818-PLATFORM_FOUNDATION-01 | PLATFORM_FOUNDATION | HIGH | ARCHITECTURE DEFECT / PROOF | ACCEPTED | TL-FIX-A | — | `docs/project/architecture/PLATFORM_FOUNDATION.md` — [Protocol v2 platform foundation target invariants (2026-08-18)](#protocol-v2-platform-foundation-target-invariants-2026-08-18) | `docs/project/maintainers/plans/PLATFORM_FOUNDATION.md` — TL-FIX-A / §6.1ax PF-TIER-ENFORCEMENT | — | — | operator accepted 2026-08-19 |
| AUDIT-20260818-PLATFORM_FOUNDATION-02 | PLATFORM_FOUNDATION | HIGH | IMPLEMENTATION DEFECT / PROOF | ACCEPTED | PF-PROOF-INTEGRITY | — | `docs/project/architecture/PLATFORM_FOUNDATION.md` — [Protocol v2 platform foundation target invariants (2026-08-18)](#protocol-v2-platform-foundation-target-invariants-2026-08-18) | `docs/project/maintainers/plans/PLATFORM_FOUNDATION.md` — PF-PROOF-INTEGRITY | — | — | operator accepted 2026-08-19 |
| AUDIT-20260818-PLATFORM_FOUNDATION-03 | PLATFORM_FOUNDATION | HIGH | IMPLEMENTATION DEFECT / TEST GAP | ACCEPTED | PF-PROOF-INTEGRITY | — | `docs/project/architecture/PLATFORM_FOUNDATION.md` — [Protocol v2 platform foundation target invariants (2026-08-18)](#protocol-v2-platform-foundation-target-invariants-2026-08-18) | `docs/project/maintainers/plans/PLATFORM_FOUNDATION.md` — PF-PROOF-INTEGRITY | — | — | operator accepted 2026-08-19 |
| AUDIT-20260818-PLATFORM_FOUNDATION-04 | PLATFORM_FOUNDATION | HIGH | RELIABILITY / CI / PROOF | ACCEPTED | TL-FIX-A | — | `docs/project/architecture/PLATFORM_FOUNDATION.md` — [Protocol v2 platform foundation target invariants (2026-08-18)](#protocol-v2-platform-foundation-target-invariants-2026-08-18) | `docs/project/maintainers/plans/PLATFORM_FOUNDATION.md` — TL-FIX-A / §6.1ax PF-TIER-ENFORCEMENT | — | — | operator accepted 2026-08-19 |
| AUDIT-20260818-PLATFORM_FOUNDATION-05 | PLATFORM_FOUNDATION | HIGH | IMPLEMENTATION/ARCHITECTURE DRIFT | ACCEPTED | PF-PROOF-INTEGRITY | — | `docs/project/architecture/PLATFORM_FOUNDATION.md` — [Protocol v2 platform foundation target invariants (2026-08-18)](#protocol-v2-platform-foundation-target-invariants-2026-08-18) | `docs/project/maintainers/plans/PLATFORM_FOUNDATION.md` — PF-PROOF-INTEGRITY | — | — | operator accepted 2026-08-19 |
| AUDIT-20260818-PLATFORM_FOUNDATION-06 | PLATFORM_FOUNDATION | LOW | LEGACY / CONTRACT CLEANLINESS | ACCEPTED | TL-FIX-A | — | `docs/project/architecture/PLATFORM_FOUNDATION.md` — [Protocol v2 platform foundation target invariants (2026-08-18)](#protocol-v2-platform-foundation-target-invariants-2026-08-18) | `docs/project/maintainers/plans/PLATFORM_FOUNDATION.md` — §6.1ax PF-TIER-ENFORCEMENT deliverable H | — | — | operator accepted 2026-08-19 |
| AUDIT-20260818-ORCHESTRATION-01 | ORCHESTRATION | HIGH | IMPLEMENTATION DEFECT / CONTRACT DEFECT | ACCEPTED | ORCH-CONTRACT-INTEGRITY | — | `docs/project/architecture/ORCHESTRATION.md` — [Protocol v2 orchestration target invariants (2026-08-18)](#protocol-v2-orchestration-target-invariants-2026-08-18) | `docs/project/maintainers/plans/ORCHESTRATION.md` — ORCH-CONTRACT-INTEGRITY | — | — | operator accepted 2026-08-20 |
| AUDIT-20260818-ORCHESTRATION-02 | ORCHESTRATION | HIGH | CONTRACT DEFECT / FAIL-OPEN CONFIGURATION | ACCEPTED | ORCH-CONTRACT-INTEGRITY | — | `docs/project/architecture/ORCHESTRATION.md` — [Protocol v2 orchestration target invariants (2026-08-18)](#protocol-v2-orchestration-target-invariants-2026-08-18) | `docs/project/maintainers/plans/ORCHESTRATION.md` — ORCH-CONTRACT-INTEGRITY | — | — | operator accepted 2026-08-20 |
| AUDIT-20260818-ORCHESTRATION-03 | ORCHESTRATION | HIGH | IMPLEMENTATION DEFECT / GRAPH SEMANTICS | ACCEPTED | ORCH-DELEGATION-INTEGRITY | — | `docs/project/architecture/ORCHESTRATION.md` — [Protocol v2 orchestration target invariants (2026-08-18)](#protocol-v2-orchestration-target-invariants-2026-08-18) | `docs/project/maintainers/plans/ORCHESTRATION.md` — ORCH-DELEGATION-INTEGRITY | — | — | operator accepted 2026-08-20 |
| AUDIT-20260818-ORCHESTRATION-04 | ORCHESTRATION | HIGH | ARCHITECTURE DEFECT / DUPLICATE CONTRACT | ACCEPTED | ORCH-CONTRACT-INTEGRITY | — | `docs/project/architecture/ORCHESTRATION.md` — [Protocol v2 orchestration target invariants (2026-08-18)](#protocol-v2-orchestration-target-invariants-2026-08-18) | `docs/project/maintainers/plans/ORCHESTRATION.md` — ORCH-CONTRACT-INTEGRITY | — | — | operator accepted 2026-08-20 |
| AUDIT-20260818-ORCHESTRATION-05 | ORCHESTRATION | MEDIUM | VALIDATION GAP / FAIL-LATE | ACCEPTED | ORCH-CONTRACT-INTEGRITY | — | `docs/project/architecture/ORCHESTRATION.md` — [Protocol v2 orchestration target invariants (2026-08-18)](#protocol-v2-orchestration-target-invariants-2026-08-18) | `docs/project/maintainers/plans/ORCHESTRATION.md` — ORCH-CONTRACT-INTEGRITY | — | — | operator accepted 2026-08-20 |
| AUDIT-20260818-AGENT_SYSTEM-01 | AGENT_SYSTEM | HIGH | IMPLEMENTATION DEFECT / GOVERNANCE BYPASS | ACCEPTED | AGSYS-CONTRACT-INTEGRITY | — | `docs/project/architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md` — [Protocol v2 agent system target invariants (2026-08-18)](#protocol-v2-agent-system-target-invariants-2026-08-18) | `docs/project/maintainers/plans/AGENT_CONTRACTS_AND_ASSEMBLY.md` — AGSYS-CONTRACT-INTEGRITY | — | — | operator accepted 2026-08-20 |
| AUDIT-20260818-AGENT_SYSTEM-02 | AGENT_SYSTEM | HIGH | CONTRACT / STATE INTEGRITY DEFECT | ACCEPTED | AGSYS-CONTRACT-INTEGRITY | — | `docs/project/architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md` — [Protocol v2 agent system target invariants (2026-08-18)](#protocol-v2-agent-system-target-invariants-2026-08-18) | `docs/project/maintainers/plans/AGENT_CONTRACTS_AND_ASSEMBLY.md` — AGSYS-CONTRACT-INTEGRITY | — | — | operator accepted 2026-08-20 |
| AUDIT-20260818-AGENT_SYSTEM-03 | AGENT_SYSTEM | HIGH | CONTRACT / PERMISSION BOUNDARY DEFECT | ACCEPTED | AGSYS-CONTRACT-INTEGRITY | — | `docs/project/architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md` — [Protocol v2 agent system target invariants (2026-08-18)](#protocol-v2-agent-system-target-invariants-2026-08-18) | `docs/project/maintainers/plans/AGENT_CONTRACTS_AND_ASSEMBLY.md` — AGSYS-CONTRACT-INTEGRITY | — | — | operator accepted 2026-08-20 |
| AUDIT-20260818-AGENT_SYSTEM-04 | AGENT_SYSTEM | HIGH | IDENTITY / ARCHITECTURE DEFECT | ACCEPTED | AGSYS-IDENTITY-PROJECTION | — | `docs/project/architecture/AGENT_DISTRIBUTION.md` — [Protocol v2 agent system identity projection invariants (2026-08-18)](#protocol-v2-agent-system-identity-projection-invariants-2026-08-18) | `docs/project/maintainers/plans/AGENT_DISTRIBUTION.md` — AGSYS-IDENTITY-PROJECTION | — | — | operator accepted 2026-08-20 |
| AUDIT-20260818-AGENT_SYSTEM-05 | AGENT_SYSTEM | HIGH | CONTRACT DEFECT / FAIL-OPEN CONFIGURATION | ACCEPTED | AGSYS-CONTRACT-INTEGRITY | — | `docs/project/architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md` — [Protocol v2 agent system target invariants (2026-08-18)](#protocol-v2-agent-system-target-invariants-2026-08-18) | `docs/project/maintainers/plans/AGENT_CONTRACTS_AND_ASSEMBLY.md` — AGSYS-CONTRACT-INTEGRITY | — | — | operator accepted 2026-08-20 |
| AUDIT-20260818-AGENT_SYSTEM-06 | AGENT_SYSTEM | MEDIUM | IMPLEMENTATION / DOCUMENTATION DRIFT | ACCEPTED | AGSYS-CONTRACT-INTEGRITY | — | `docs/project/architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md` — [Protocol v2 agent system target invariants (2026-08-18)](#protocol-v2-agent-system-target-invariants-2026-08-18) | `docs/project/maintainers/plans/AGENT_CONTRACTS_AND_ASSEMBLY.md` — AGSYS-CONTRACT-INTEGRITY | — | — | operator accepted 2026-08-20 |
| AUDIT-20260818-TOOLS-01 | TOOLS | HIGH | ARCHITECTURE / AUTHORIZATION DEFECT | ACCEPTED | TOOLS-GOVERNED-BOUNDARY-INTEGRITY | — | `docs/project/architecture/TOOLS.md` — [Protocol v2 tools target invariants (2026-08-18)](#protocol-v2-tools-target-invariants-2026-08-18) | `docs/project/maintainers/plans/TOOLS.md` — TOOLS-GOVERNED-BOUNDARY-INTEGRITY | — | — | operator accepted 2026-08-20 |
| AUDIT-20260818-TOOLS-02 | TOOLS | HIGH | IMPLEMENTATION DEFECT / RESOURCE BOUNDARY | ACCEPTED | TOOLS-GOVERNED-BOUNDARY-INTEGRITY | — | `docs/project/architecture/TOOLS.md` — [Protocol v2 tools target invariants (2026-08-18)](#protocol-v2-tools-target-invariants-2026-08-18) | `docs/project/maintainers/plans/TOOLS.md` — TOOLS-GOVERNED-BOUNDARY-INTEGRITY | — | — | operator accepted 2026-08-20 |
| AUDIT-20260818-TOOLS-03 | TOOLS | HIGH | IMPLEMENTATION DEFECT / BUDGET GOVERNANCE | ACCEPTED | TOOLS-GOVERNED-BOUNDARY-INTEGRITY | — | `docs/project/architecture/TOOLS.md` — [Protocol v2 tools target invariants (2026-08-18)](#protocol-v2-tools-target-invariants-2026-08-18) | `docs/project/maintainers/plans/TOOLS.md` — TOOLS-GOVERNED-BOUNDARY-INTEGRITY | — | — | operator accepted 2026-08-20 |
| AUDIT-20260818-TOOLS-04 | TOOLS | HIGH | IDENTITY / IDEMPOTENCY DEFECT | ACCEPTED | TOOLS-SIDE-EFFECT-SAFETY | — | `docs/project/architecture/TOOLS.md` — [Protocol v2 tools target invariants (2026-08-18)](#protocol-v2-tools-target-invariants-2026-08-18) | `docs/project/maintainers/plans/TOOLS.md` — TOOLS-SIDE-EFFECT-SAFETY | — | — | operator accepted 2026-08-20 |
| AUDIT-20260818-TOOLS-05 | TOOLS | HIGH | SIDE-EFFECT / RETRY ARCHITECTURE DEFECT | ACCEPTED | TOOLS-SIDE-EFFECT-SAFETY | — | `docs/project/architecture/TOOLS.md` — [Protocol v2 tools target invariants (2026-08-18)](#protocol-v2-tools-target-invariants-2026-08-18) | `docs/project/maintainers/plans/TOOLS.md` — TOOLS-SIDE-EFFECT-SAFETY | — | — | operator accepted 2026-08-20 |
| AUDIT-20260818-TOOLS-06 | TOOLS | MEDIUM | FAILURE / IDEMPOTENCY STATE MODEL GAP | ACCEPTED | TOOLS-SIDE-EFFECT-SAFETY | — | `docs/project/architecture/TOOLS.md` — [Protocol v2 tools target invariants (2026-08-18)](#protocol-v2-tools-target-invariants-2026-08-18) | `docs/project/maintainers/plans/TOOLS.md` — TOOLS-SIDE-EFFECT-SAFETY | — | — | operator accepted 2026-08-20 |
| AUDIT-20260818-SKILLS-01 | SKILLS | HIGH | AUTHORIZATION / HOST-BOUNDARY DEFECT | ACCEPTED | SKILLS-AUTHORITY-INTEGRITY | — | `docs/project/architecture/SKILLS.md` — [Protocol v2 skills target invariants (2026-08-18)](#protocol-v2-skills-target-invariants-2026-08-18) | `docs/project/maintainers/plans/SKILLS.md` — SKILLS-AUTHORITY-INTEGRITY | — | — | operator accepted 2026-08-20 |
| AUDIT-20260818-SKILLS-02 | SKILLS | HIGH | CONTRACT / VERSION IDENTITY DEFECT | ACCEPTED | SKILLS-IDENTITY-PROVENANCE | — | `docs/project/architecture/SKILLS.md` — [Protocol v2 skills target invariants (2026-08-18)](#protocol-v2-skills-target-invariants-2026-08-18) | `docs/project/maintainers/plans/SKILLS.md` — SKILLS-IDENTITY-PROVENANCE | — | — | operator accepted 2026-08-20 |
| AUDIT-20260818-SKILLS-03 | SKILLS | HIGH | AUTHORIZATION / CAPABILITY EXPANSION DEFECT | ACCEPTED | SKILLS-AUTHORITY-INTEGRITY | — | `docs/project/architecture/SKILLS.md` — [Protocol v2 skills target invariants (2026-08-18)](#protocol-v2-skills-target-invariants-2026-08-18) | `docs/project/maintainers/plans/SKILLS.md` — SKILLS-AUTHORITY-INTEGRITY | — | — | operator accepted 2026-08-20; cross-ref TOOLS-01 monotonic authority |
| AUDIT-20260818-SKILLS-04 | SKILLS | MEDIUM | PROVENANCE / CONTRACT COMPLETENESS GAP | ACCEPTED | SKILLS-IDENTITY-PROVENANCE | — | `docs/project/architecture/SKILLS.md` — [Protocol v2 skills target invariants (2026-08-18)](#protocol-v2-skills-target-invariants-2026-08-18) | `docs/project/maintainers/plans/SKILLS.md` — SKILLS-IDENTITY-PROVENANCE | — | — | operator accepted 2026-08-20 |
| AUDIT-20260818-SKILLS-05 | SKILLS | MEDIUM | CONFIGURATION VALIDATION GAP | ACCEPTED | SKILLS-AUTHORITY-INTEGRITY | — | `docs/project/architecture/SKILLS.md` — [Protocol v2 skills target invariants (2026-08-18)](#protocol-v2-skills-target-invariants-2026-08-18) | `docs/project/maintainers/plans/SKILLS.md` — SKILLS-AUTHORITY-INTEGRITY | — | — | operator accepted 2026-08-20 |
| AUDIT-20260818-SKILLS-06 | SKILLS | MEDIUM | DOCUMENTATION / EVIDENCE DRIFT | ACCEPTED | SKILLS-EVIDENCE-SYNC | — | `docs/project/architecture/SKILLS.md` — [Protocol v2 skills target invariants (2026-08-18)](#protocol-v2-skills-target-invariants-2026-08-18) | `docs/project/maintainers/plans/SKILLS.md` — SKILLS-EVIDENCE-SYNC | — | — | operator accepted 2026-08-20 |
| AUDIT-20260818-CODE_CRAFT-01 | CODE_CRAFT | CRITICAL | SECURITY / TENANT ISOLATION / IDENTITY DEFECT | ACCEPTED | CODECRAFT-IDENTITY-GOVERNANCE-INTEGRITY | — | `docs/project/architecture/CODE_CRAFT.md` — [Protocol v2 CodeCraft target invariants (2026-08-18)](#protocol-v2-codecraft-target-invariants-2026-08-18) | `docs/project/maintainers/plans/CODE_CRAFT.md` — CODECRAFT-IDENTITY-GOVERNANCE-INTEGRITY | — | — | operator accepted 2026-08-20 |
| AUDIT-20260818-CODE_CRAFT-02 | CODE_CRAFT | CRITICAL | GOVERNANCE / HITL AUTHORIZATION BYPASS | ACCEPTED | CODECRAFT-IDENTITY-GOVERNANCE-INTEGRITY | — | `docs/project/architecture/CODE_CRAFT.md` — [Protocol v2 CodeCraft target invariants (2026-08-18)](#protocol-v2-codecraft-target-invariants-2026-08-18) | `docs/project/maintainers/plans/CODE_CRAFT.md` — CODECRAFT-IDENTITY-GOVERNANCE-INTEGRITY | — | — | operator accepted 2026-08-20 |
| AUDIT-20260818-CODE_CRAFT-03 | CODE_CRAFT | HIGH | AUTHORIZATION / CONFIGURATION ESCALATION | ACCEPTED | CODECRAFT-IDENTITY-GOVERNANCE-INTEGRITY | — | `docs/project/architecture/CODE_CRAFT.md` — [Protocol v2 CodeCraft target invariants (2026-08-18)](#protocol-v2-codecraft-target-invariants-2026-08-18) | `docs/project/maintainers/plans/CODE_CRAFT.md` — CODECRAFT-IDENTITY-GOVERNANCE-INTEGRITY | — | — | operator accepted 2026-08-20 |
| AUDIT-20260818-CODE_CRAFT-04 | CODE_CRAFT | HIGH | VERIFICATION / PROMOTION INTEGRITY DEFECT | ACCEPTED | CODECRAFT-VERIFICATION-INTEGRITY | — | `docs/project/architecture/CODE_CRAFT.md` — [Protocol v2 CodeCraft target invariants (2026-08-18)](#protocol-v2-codecraft-target-invariants-2026-08-18) | `docs/project/maintainers/plans/CODE_CRAFT.md` — CODECRAFT-VERIFICATION-INTEGRITY | — | — | operator accepted 2026-08-20 |
| AUDIT-20260818-CODE_CRAFT-05 | CODE_CRAFT | HIGH | VERIFICATION / ISOLATION DEFECT | ACCEPTED | CODECRAFT-VERIFICATION-INTEGRITY | — | `docs/project/architecture/CODE_CRAFT.md` — [Protocol v2 CodeCraft target invariants (2026-08-18)](#protocol-v2-codecraft-target-invariants-2026-08-18) | `docs/project/maintainers/plans/CODE_CRAFT.md` — CODECRAFT-VERIFICATION-INTEGRITY | — | — | operator accepted 2026-08-20 |
| AUDIT-20260818-CODE_CRAFT-06 | CODE_CRAFT | HIGH | ARCHITECTURE / SECURITY ISOLATION DEFECT | ACCEPTED | CODECRAFT-ISOLATION-INTEGRITY | — | `docs/project/architecture/CODE_CRAFT.md` — [Protocol v2 CodeCraft target invariants (2026-08-18)](#protocol-v2-codecraft-target-invariants-2026-08-18) | `docs/project/maintainers/plans/CODE_CRAFT.md` — CODECRAFT-ISOLATION-INTEGRITY | — | — | operator accepted 2026-08-20 |
| AUDIT-20260818-CODE_CRAFT-07 | CODE_CRAFT | HIGH | SECURITY / PAPER CONTROL | ACCEPTED | CODECRAFT-ISOLATION-INTEGRITY | — | `docs/project/architecture/CODE_CRAFT.md` — [Protocol v2 CodeCraft target invariants (2026-08-18)](#protocol-v2-codecraft-target-invariants-2026-08-18) | `docs/project/maintainers/plans/CODE_CRAFT.md` — CODECRAFT-ISOLATION-INTEGRITY | — | — | operator accepted 2026-08-20 |
## Audit rollup

**Status:** pending — campaign `IN_PROGRESS`; frozen at audit `COMPLETE` only.

**Completed layers:** 15

First layer summary:

- **Layer:** STRATEGIC_HARNESS_MODEL — **FAIL** at `9658224495c775fcefd55ab52bbcc7a94c84fb50`
- **Accepted findings:** 10 total — 0 CRITICAL, 6 HIGH, 4 MEDIUM, 0 LOW
- **Systemic themes:** universal governed execution boundary; identity/typed author surface; production host neutrality; maturity claims vs verified invariants
- **Recommended remediation order:** SHM-FIX-A → SHM-FIX-B → SHM-FIX-C → SHM-FIX-D (see layer report)

Second layer summary:

- **Layer:** TIER_LAYER_BOUNDARIES — **FAIL** at `d8d10bb5099d003eb9495674c28e0f6e6762dbfa`
- **Accepted findings:** 5 total — 0 CRITICAL, 2 HIGH, 3 MEDIUM, 0 LOW
- **Systemic themes:** executable tier ownership proof; single agent identity authority; product-neutral Tier-3 contracts; public application composition API; consumer static-contract coverage
- **Recommended remediation order:** TL-FIX-A → TL-FIX-B → TL-FIX-C → TL-FIX-D (see layer report)

Third layer summary:

- **Layer:** PROVIDER_BACKEND_ABSTRACTION — **FAIL** at `7570e9b4508554a42bdf5cce2c987c56c6f2b80e`
- **Accepted findings:** 5 total — 0 CRITICAL, 2 HIGH, 3 MEDIUM, 0 LOW
- **Systemic themes:** concrete persistence leakage through paper abstractions; canonical observability provider bypass; provider-specific configuration leakage; incomplete vendor-boundary proof; missing Experimentation persistence abstraction
- **Recommended remediation order:** PBA-FIX-A → PBA-FIX-B → PBA-FIX-C → PBA-FIX-D (see layer report)

Fourth layer summary:

- **Layer:** INTERFACE_TASK_INTAKE — **FAIL** at `2640d826da6f1a781e798326ff1b21b3a9f7c4cc`
- **Accepted findings:** 6 total — 0 CRITICAL, 3 HIGH, 3 MEDIUM, 0 LOW
- **Systemic themes:** canonical normalized intake contract adoption; distinct TaskId/RunId on public surfaces; UnifiedTaskRunner convergence before Nexus; typed intake semantics preservation; typed executor interfaces; E2E streaming intake parity proof
- **Recommended remediation order:** ITI-FIX-A → ITI-FIX-B → ITI-FIX-C → ITI-FIX-D (see layer report)

Fifth layer summary:

- **Layer:** IDENTITY_TRUST — **FAIL** at `6fbc5e4928963ecd386456158b0753662fed209b`
- **Accepted findings:** 6 total — 0 CRITICAL, 4 HIGH, 2 MEDIUM, 0 LOW
- **Systemic themes:** authenticated principal spine; delegated authority enforcement; human approver provenance; resume surface HITL correlation; execution identity closure on residual paths; actor/principal model coherence
- **Recommended remediation order:** IDT-FIX-A → IDT-FIX-B → IDT-FIX-C → IDT-FIX-D (see layer report)


Sixth layer summary:

- **Layer:** POLICY_GOVERNANCE — **FAIL** at `042cc9b50386cfcd4da30310c84d000dbf5d2718`
- **Accepted findings:** 5 total — 0 CRITICAL, 4 HIGH, 1 MEDIUM, 0 LOW
- **Systemic themes:** canonical side-effect spine; policy precedence; scoped approval consumption; explicit matching
- **Recommended remediation order:** PG-FIX-A → PG-FIX-B → PG-FIX-C → PG-FIX-D (see layer report)

Seventh layer summary:

- **Layer:** LLM_ADAPTERS — **FAIL** at `b1e4de1d776acc64e8461f7dcdce09cd03d07b80`
- **Accepted findings:** 6 total — 0 CRITICAL, 4 HIGH, 2 MEDIUM, 0 LOW
- **Systemic themes:** universal PRE_MODEL boundary; decision-to-execution binding; governed failover; LLM identity closure
- **Recommended remediation order:** LLM-FIX-A → LLM-FIX-B → LLM-FIX-C → LLM-FIX-D (see layer report)

Eighth layer summary:

- **Layer:** REASONING_PLANNING — **FAIL** at `fe876d301df07ce22e438b0a55167275ccec32b5`
- **Accepted findings:** 6 total — 0 CRITICAL, 4 HIGH, 2 MEDIUM, 0 LOW
- **Systemic themes:** plan integrity; production eligibility parity; replan closure; cognitive verdict integrity; tool-plan semantics; product-neutral planner core
- **Recommended remediation order:** RPL-FIX-A → RPL-FIX-B → RPL-FIX-C → RPL-FIX-D → RPL-FIX-E → RPL-FIX-F (see layer report)

Ninth layer summary:

- **Layer:** EXECUTION_RUNTIME — **FAIL** at `df7aaac19b20e84c06d6233492cdb4365a892f4f`
- **Accepted findings:** 6 total — 0 CRITICAL, 5 HIGH, 1 MEDIUM, 0 LOW
- **Systemic themes:** canonical runtime policy propagation; atomic step commits; attempt continuity; exception containment; cooperative cancellation and checkpoint invalidation
- **Recommended remediation order:** UER-FIX-A → UER-FIX-B → UER-FIX-C → UER-FIX-D → UER-FIX-E (see layer report)

Tenth layer summary:

- **Layer:** PLATFORM_FOUNDATION — **FAIL** at `f21d5c3dc417907acb50d597642d3892e704bd47`
- **Accepted findings:** 6 total — 0 CRITICAL, 5 HIGH, 0 MEDIUM, 1 LOW
- **Systemic themes:** authoritative tier enforcement proof; fail-closed foundation proof runners; integration-path protection; gate-contract parity
- **Recommended remediation order:** TL-FIX-A → PF-PROOF-INTEGRITY (see layer report)

Eleventh layer summary:

- **Layer:** ORCHESTRATION — **FAIL** at `a784966681782bc58412af290c2978c1d1f152a3`
- **Accepted findings:** 5 total — 0 CRITICAL, 4 HIGH, 1 MEDIUM, 0 LOW
- **Systemic themes:** canonical graph-node executable identity; typed fail-fast orchestration configuration; exact delegation-edge provenance; single OrchestrationProfile ownership; static graph cycle rejection
- **Recommended remediation order:** ORCH-CONTRACT-INTEGRITY → ORCH-DELEGATION-INTEGRITY (see layer report)

Twelfth layer summary:

- **Layer:** AGENT_SYSTEM — **FAIL** at `654a7c0e3fe823a43a2620645848248023e1c64e`
- **Accepted findings:** 6 total — 0 CRITICAL, 5 HIGH, 1 MEDIUM, 0 LOW
- **Systemic themes:** production eligibility gate inversion; mutable registered contract; allowed_tools bypass; registry identity rewrite; fail-open contract schema; on-call enforcement drift
- **Recommended remediation order:** AGSYS-CONTRACT-INTEGRITY → AGSYS-IDENTITY-PROJECTION (see layer report)

Thirteenth layer summary:

- **Layer:** TOOLS — **FAIL** at `65aaf33a6a6dba9b336162ec547cd677f4edad91`
- **Accepted findings:** 6 total — 0 CRITICAL, 5 HIGH, 1 MEDIUM, 0 LOW
- **Systemic themes:** monotonic permission intersection; real timeout boundary vs external cancellation limits; pre-invoke hard budget accounting; canonical idempotency operation identity; side-effect retry authorization; idempotency outcome-state semantics
- **Recommended remediation order:** TOOLS-GOVERNED-BOUNDARY-INTEGRITY → TOOLS-SIDE-EFFECT-SAFETY (see layer report)

Fourteenth layer summary:

- **Layer:** SKILLS — **FAIL** at `2df2f07d10aa19c4d62694f21858be501a3d6d18`
- **Accepted findings:** 6 total — 0 CRITICAL, 3 HIGH, 3 MEDIUM, 0 LOW
- **Systemic themes:** fail-closed host Skill authority; explicit Skill version identity; non-expanding ToolProfile authority; canonical ResolvedSkillPack provenance; fail-fast SkillProfile references; catalog-count evidence sync
- **Recommended remediation order:** SKILLS-AUTHORITY-INTEGRITY → SKILLS-IDENTITY-PROVENANCE → SKILLS-EVIDENCE-SYNC (see layer report)

Fifteenth layer summary:

- **Layer:** CODE_CRAFT — **FAIL** at `f985ad342d0d6db38c9998df67f9cd7bc10bfa46`
- **Accepted findings:** 7 total — 2 CRITICAL, 5 HIGH, 0 MEDIUM, 0 LOW
- **Systemic themes:** canonical execution identity binding for craft sessions; HITL approval from Governed Execution not tool input; narrow-only task override lattice; evidence-consuming promotion; same-sandbox verification; isolation anti-downgrade; runtime network egress enforcement
- **Recommended remediation order:** CODECRAFT-IDENTITY-GOVERNANCE-INTEGRITY → CODECRAFT-VERIFICATION-INTEGRITY → CODECRAFT-ISOLATION-INTEGRITY (see layer report)

**Cumulative (completed layers only):** 85 accepted findings — 2 CRITICAL, 56 HIGH, 27 MEDIUM, 0 LOW

## Remediation rollup

**Status:** not started — campaign audit still `IN_PROGRESS`; normal remediation queue builds after campaign `COMPLETE` unless operator scopes work to named findings.

| remediation_block | findings | status | notes |
|-------------------|----------|--------|-------|
| SHM-FIX-A | AUDIT-20260818-STRATEGIC_HARNESS_MODEL-01, 02, 03, 04 | ACCEPTED / PLANNED | execution boundary — not implemented in this persistence task |
| SHM-FIX-B | AUDIT-20260818-STRATEGIC_HARNESS_MODEL-06, 08, 09 | ACCEPTED / PLANNED | identity and typed context |
| SHM-FIX-C | AUDIT-20260818-STRATEGIC_HARNESS_MODEL-05, 07 | ACCEPTED / PLANNED | host and platform neutrality |
| SHM-FIX-D | AUDIT-20260818-STRATEGIC_HARNESS_MODEL-10 | ACCEPTED / PLANNED | maturity recertification after A–C verification |
| TL-FIX-A | AUDIT-20260818-TIER_LAYER_BOUNDARIES-01, 05; AUDIT-20260818-PLATFORM_FOUNDATION-01, 04 | ACCEPTED / PLANNED | executable tier ownership + integration-path protection — not implemented in this persistence task |
| TL-FIX-B | AUDIT-20260818-TIER_LAYER_BOUNDARIES-02 | ACCEPTED / PLANNED | single agent ownership |
| TL-FIX-C | AUDIT-20260818-TIER_LAYER_BOUNDARIES-03 | ACCEPTED / PLANNED | product-neutral Tier-3 platform |
| TL-FIX-D | AUDIT-20260818-TIER_LAYER_BOUNDARIES-04 | ACCEPTED / PLANNED | public application composition contract |
| PBA-FIX-A | AUDIT-20260818-PROVIDER_BACKEND_ABSTRACTION-01 | ACCEPTED / PLANNED | long-running checkpoint port consumption — not implemented in this persistence task |
| PBA-FIX-B | AUDIT-20260818-PROVIDER_BACKEND_ABSTRACTION-02, 04 | ACCEPTED / PLANNED | observability export boundary + vendor-boundary governance |
| PBA-FIX-C | AUDIT-20260818-PROVIDER_BACKEND_ABSTRACTION-03 | ACCEPTED / PLANNED | provider-owned guardrail configuration |
| PBA-FIX-D | AUDIT-20260818-PROVIDER_BACKEND_ABSTRACTION-05 | ACCEPTED / PLANNED | experiment persistence port — not implemented in this persistence task |
| ITI-FIX-A | AUDIT-20260818-INTERFACE_TASK_INTAKE-01, 04 | ACCEPTED / PLANNED | canonical normalized intake contract — not implemented in this persistence task |
| ITI-FIX-B | AUDIT-20260818-INTERFACE_TASK_INTAKE-02 | ACCEPTED / PLANNED | distinct TaskId/RunId on public intake paths |
| ITI-FIX-C | AUDIT-20260818-INTERFACE_TASK_INTAKE-03, 05 | ACCEPTED / PLANNED | UnifiedTaskRunner convergence + typed executor interface |
| ITI-FIX-D | AUDIT-20260818-INTERFACE_TASK_INTAKE-06 | ACCEPTED / PLANNED | E2E streaming intake parity proof after A/B/C |
| IDT-FIX-A | AUDIT-20260818-IDENTITY_TRUST-01, 06 | ACCEPTED / PLANNED | authenticated principal spine — not implemented in this persistence task |
| IDT-FIX-B | AUDIT-20260818-IDENTITY_TRUST-02 | ACCEPTED / PLANNED | delegated authority narrowing |
| IDT-FIX-C | AUDIT-20260818-IDENTITY_TRUST-03, 04 | ACCEPTED / PLANNED | human decision provenance + resume surface alignment |
| IDT-FIX-D | AUDIT-20260818-IDENTITY_TRUST-05 | ACCEPTED / PLANNED | execution identity closure on residual HITL/lifecycle paths |

| PG-FIX-A | AUDIT-20260818-POLICY_GOVERNANCE-01, 03 | ACCEPTED / PLANNED | canonical side-effect spine — not implemented in this persistence task |
| PG-FIX-B | AUDIT-20260818-POLICY_GOVERNANCE-02 | ACCEPTED / PLANNED | safe policy resolution semantics |
| PG-FIX-C | AUDIT-20260818-POLICY_GOVERNANCE-04 | ACCEPTED / PLANNED | scoped approval consumption |
| PG-FIX-D | AUDIT-20260818-POLICY_GOVERNANCE-05 | ACCEPTED / PLANNED | explicit policy matching |
| LLM-FIX-A | AUDIT-20260818-LLM_ADAPTERS-01, 04 | ACCEPTED / PLANNED | universal inference boundary — not implemented in this persistence task |
| LLM-FIX-B | AUDIT-20260818-LLM_ADAPTERS-02, 05 | ACCEPTED / PLANNED | decision-to-execution binding |
| LLM-FIX-C | AUDIT-20260818-LLM_ADAPTERS-03 | ACCEPTED / PLANNED | governed failover |
| LLM-FIX-D | AUDIT-20260818-LLM_ADAPTERS-06 | ACCEPTED / PLANNED | LLM execution identity closure; cross-ref IDT-FIX-D |
| RPL-FIX-A | AUDIT-20260818-REASONING_PLANNING-01 | ACCEPTED / PLANNED | canonical plan integrity — not implemented in this persistence task |
| RPL-FIX-B | AUDIT-20260818-REASONING_PLANNING-02 | ACCEPTED / PLANNED | planning/execution eligibility parity |
| RPL-FIX-C | AUDIT-20260818-REASONING_PLANNING-03 | ACCEPTED / PLANNED | replan semantic closure |
| RPL-FIX-D | AUDIT-20260818-REASONING_PLANNING-04 | ACCEPTED / PLANNED | cognitive verdict integrity |
| RPL-FIX-E | AUDIT-20260818-REASONING_PLANNING-05 | ACCEPTED / PLANNED | tool-planning outcome semantics |
| RPL-FIX-F | AUDIT-20260818-REASONING_PLANNING-06 | ACCEPTED / PLANNED | remove product-shaped core planning |
| UER-FIX-A | AUDIT-20260818-EXECUTION_RUNTIME-01 | ACCEPTED / PLANNED | canonical runtime policy propagation — not implemented in this persistence task |
| UER-FIX-B | AUDIT-20260818-EXECUTION_RUNTIME-02 | ACCEPTED / PLANNED | atomic step commit semantics |
| UER-FIX-C | AUDIT-20260818-EXECUTION_RUNTIME-03 | ACCEPTED / PLANNED | resume identity continuity |
| UER-FIX-D | AUDIT-20260818-EXECUTION_RUNTIME-04 | ACCEPTED / PLANNED | runtime exception containment |
| UER-FIX-E | AUDIT-20260818-EXECUTION_RUNTIME-05, 06 | ACCEPTED / PLANNED | cooperative cancellation and checkpoint invalidation |
| PF-PROOF-INTEGRITY | AUDIT-20260818-PLATFORM_FOUNDATION-02, 03, 05 | ACCEPTED / PLANNED | foundation proof runners and CI/docs gate-contract parity — not implemented in this persistence task |
| ORCH-CONTRACT-INTEGRITY | AUDIT-20260818-ORCHESTRATION-01, 02, 04, 05 | ACCEPTED / PLANNED | graph identity, typed fail-fast config, canonical OrchestrationProfile, static cycle validation — not implemented in this persistence task |
| ORCH-DELEGATION-INTEGRITY | AUDIT-20260818-ORCHESTRATION-03 | ACCEPTED / PLANNED | exact delegation-edge provenance and multi-parent policy — not implemented in this persistence task |
| AGSYS-CONTRACT-INTEGRITY | AUDIT-20260818-AGENT_SYSTEM-01, 02, 03, 05, 06 | ACCEPTED / PLANNED | production eligibility, immutable contract, canonical tool resolution, fail-fast schema, on-call parity — not implemented in this persistence task |
| AGSYS-IDENTITY-PROJECTION | AUDIT-20260818-AGENT_SYSTEM-04 | ACCEPTED / PLANNED | registry bootstrap canonical identity preservation; cross-ref TL-FIX-B — not implemented in this persistence task |
| TOOLS-GOVERNED-BOUNDARY-INTEGRITY | AUDIT-20260818-TOOLS-01, 02, 03 | ACCEPTED / PLANNED | permission intersection, effective timeout, pre-invoke budget — not implemented in this persistence task |
| TOOLS-SIDE-EFFECT-SAFETY | AUDIT-20260818-TOOLS-04, 05, 06 | ACCEPTED / PLANNED | idempotency operation identity, retry safety, outcome-state model — not implemented in this persistence task |
| SKILLS-AUTHORITY-INTEGRITY | AUDIT-20260818-SKILLS-01, 03, 05 | ACCEPTED / PLANNED | fail-closed host Skill authority, non-expanding ToolProfile, fail-fast profile references — not implemented in this persistence task |
| SKILLS-IDENTITY-PROVENANCE | AUDIT-20260818-SKILLS-02, 04 | ACCEPTED / PLANNED | explicit Skill version identity, canonical ResolvedSkillPack provenance — not implemented in this persistence task |
| SKILLS-EVIDENCE-SYNC | AUDIT-20260818-SKILLS-06 | ACCEPTED / PLANNED | catalog count single source of truth (docs-owned) — not implemented in this persistence task |
| CODECRAFT-IDENTITY-GOVERNANCE-INTEGRITY | AUDIT-20260818-CODE_CRAFT-01, 02, 03 | ACCEPTED / PLANNED | session authority, canonical HITL, override lattice — not implemented in this persistence task |
| CODECRAFT-VERIFICATION-INTEGRITY | AUDIT-20260818-CODE_CRAFT-04, 05 | ACCEPTED / PLANNED | promotion eligibility/evidence, same-sandbox verification — not implemented in this persistence task |
| CODECRAFT-ISOLATION-INTEGRITY | AUDIT-20260818-CODE_CRAFT-06, 07 | ACCEPTED / PLANNED | isolation anti-downgrade, network egress enforcement — not implemented in this persistence task |
