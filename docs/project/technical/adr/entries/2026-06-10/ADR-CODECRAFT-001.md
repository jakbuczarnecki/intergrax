# ADR-CODECRAFT-001: Ephemeral Code Craft as a separate Harness domain

| Field | Value |
|-------|-------|
| **Status** | Accepted |
| **Date** | 2026-06-10 |
| **Deciders** | Harness platform |
| **Related** | [`architecture/CODE_CRAFT.md`](../../architecture/CODE_CRAFT.md) · [`plan/CODE_CRAFT.md`](../../plan/CODE_CRAFT.md) |

## Context

Advanced agentic systems need to synthesize, test, and iteratively refine executable code when catalog tools are insufficient. Intergrax already provides sandbox execution (`runtime/sandbox`, `code.exec`, `sandbox.exec`) and compositional skills (`sandbox.code_exec`, `sandbox.refactor_loop`), but agents must implement generate→test→fix loops themselves in Tier-2.

AUDIT-IDEAL-11.1 (sandboxed execution for side-effectful tools) is **Done** — it covers **single-shot** isolated execution, not harness-orchestrated ephemeral tool synthesis.

Options considered:

1. Extend **TOOLS** only — add `codecraft.*` tool_ids without a dedicated domain.
2. Extend **SKILLS** — bundle tools + prompts; no runtime orchestrator.
3. Fold into **RELIABILITY** — treat as isolation primitive alongside sandbox.
4. **New domain pair `CODE_CRAFT`** — engine + profile + plan, analogous to RAG and CRITIC_VERIFICATION.

## Decision

- Introduce **Ephemeral Code Craft (ECC)** as the **21st platform domain pair**: `architecture/CODE_CRAFT.md` ↔ `plan/CODE_CRAFT.md`.
- Implement as a **Harness engine** that **composes** existing substrates — not a parallel sandbox:
  - **Tier-0** `intergrax/codecraft` — contracts, static gate, tool providers (`codecraft.*`).
  - **Tier-1** `intergrax/runtime/codecraft` — `CodeCraftOrchestrator`, session lifecycle, UAEP/CVL integration.
  - **Execution substrate** remains `runtime/sandbox` + optional `sandbox_host` integrations (`e2b`, `modal`, `daytona`).
- Expose LLM-facing operations via catalog tools (`codecraft.start`, `codecraft.run`, …) routed through `ToolRuntime` — same pattern as `rag.*` and `eval.judge`.
- Add `CodeCraftProfile` on `ApplicationEnvironmentProfile` (Tier-3) with modes: `disabled`, `dry_run`, `assist_only`, `supervised`, `autonomous`.
- **Ephemeral tools** live in task-scoped registry only — they MUST NOT be registered in the global Tool Library catalog.
- Verification reuses **CVL** L0/L1 (`StaticCodeGate`, optional `eval.judge`) — ECC does not create a second verification system.

Rejected:

- **TOOLS-only** — conflates atomic catalog with orchestration engine; `TOOLS.md` already documents 190 tools; ECC has sessions, iterations, promotion.
- **SKILLS-only** — skills resolve at bind time and do not orchestrate runtime loops.
- **RELIABILITY-only** — sandbox isolation is substrate; ECC is synthesis + promotion semantics.

## Consequences

- Hub, AGENTS.md, audit map, and README updated to **21 domain pairs**.
- Implementation delivered in phases **ECC-0 … ECC-6** per [`plan/CODE_CRAFT.md`](../../plan/CODE_CRAFT.md).
- `TOOLS.md` retains `code.exec` / `sandbox.exec` as low-level primitives; cross-links to CODE_CRAFT for orchestrated flows.
- Production hosts default `CodeCraftProfile.mode=disabled` or `supervised`; lab may use `autonomous` with local sandbox.
- Significant security surface — cloud/container isolation tiers required before regulated profiles ship.

## Compliance

- Tier boundaries preserved: Tier-2 agents invoke `codecraft.*` via UAEP; no agent-local subprocess loops.
- No duplicate sandbox runtime — reuse `SandboxSession` / `HostedSandboxSession`.
- ADR linked from hub and CODE_CRAFT domain pair.
