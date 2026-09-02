# ADR-INTERGRAX_ASSISTANT-001: Hub-and-spoke harness chat environment

**Status:** Accepted  
**Date:** 2026-06-08  
**Domain:** Tier-3 application (`intergrax_assistant_application`)

## Context

Intergrax needs a **ChatGPT-class conversational environment** built on the Harness AI stack - not a legacy monolithic chatbot. Operators must be able to:

- run **fully local** (Ollama) or swap to any registered `LLMAdapter` via env,
- use session memory, user LTM, RAG, tools, integrations, and skills through Tier-0,
- optionally delegate to **existing Tier-2 platform agents** (Legal, Research, …) during a chat turn,
- experiment with the full Agent OS without forking Nexus.

Prior art: `lab_application` (multi-agent debug lab, no chat product shape), `legal_application` (single-domain chat SKU), `local_workspace_application` (multi-agent pipeline, not general chat). None combines **conversational hub + swappable LLM + optional specialist delegation** in one Tier-3 host.

Intergrax forbids Tier-2 agents calling other agents directly; delegation must flow through Nexus `DelegationSpec` / classifier / engine planner (§42.14.3, Appendix I).

## Decision

Introduce **Intergrax Assistant (IAA)** as a Tier-3 **lab-profile application** with a **hub-and-spoke** topology:

| Layer | Choice |
|-------|--------|
| **Tier-3 host** | `intergrax_assistant_application` - HTTP `/v1/intergrax_assistant/*`, MCP, debug API |
| **Hub agent (Tier-2)** | `intergrax_assistant` - capability `platform.assist` - default conversational entry |
| **Specialists (optional)** | Echo, Legal, Research, Summary - mounted via env flags; invoked by Nexus delegation, not direct agent calls |
| **LLM** | `LLMProfile` resolved from `INTERGRAX_LLM_*` (default `ollama` / `llama3.1:latest`) on `ApplicationEnvironmentProfile` |
| **Orchestration** | Engine planner opt-in (`INTERGRAX_ASSISTANT_ENGINE_PLANNER`); `max_delegation_depth` configurable |
| **Memory / RAG** | `ApplicationEnvironmentProfile.lab_defaults()` - harness memory + RAG + curated tool allow-list |

Default roster is **hub-only** (minimal smoke path). Specialists are opt-in (`INTERGRAX_ASSISTANT_INCLUDE_*`) to keep lab startup predictable.

## Consequences

**Positive**

- Demonstrates full Harness stack in one chat-shaped product shell
- LLM provider swap without code changes - key differentiator vs single-vendor chatbots
- Reuses platform agents as delegation targets - validates graph-native subagent model
- Clear tier hygiene: product wiring in Tier-3, conversation logic in Tier-2 hub

**Negative**

- Not a production SKU yet - lab profile, no dedicated `/chat` product routes (uses harness `/run`)
- Hub agent UAEP is scaffold baseline - domain chat loop (tool planner, file intake) is follow-up work
- Engine planner requires LLM at bootstrap - misconfigured local Ollama fails fast at first run

**Follow-up**

- IAA.2 - dedicated chat route + `session_id` contract (mirror `legal_application`)
- IAA.3 - file attachment intake + workspace RAG collections
- IAA.4 - explicit `ApplicationGraphSpec` for common delegation paths
- Product promotion (`--profile product`) only after security/HITL review

**Platform ADR:** no harness ADR needed - composition of existing Tier-0/Tier-1 mechanisms; no Nexus contract change.
