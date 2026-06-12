# intergrax_assistant agent — architecture

**Status:** Scaffold baseline (2026-06-08) — hub / concierge role defined  
**Tier:** Tier-2 (`agents/intergrax_assistant/`)  
**Host:** [`applications/intergrax_assistant_application/`](../../applications/intergrax_assistant_application/)  
**Platform ADR:** [`ADR-INTERGRAX_ASSISTANT-001`](../../applications/intergrax_assistant_application/adr/ADR-INTERGRAX_ASSISTANT-001.md)

---

## Role

**Conversational hub agent** for the Intergrax Assistant environment. Default entry for chat-shaped tasks (`platform.assist`).

| Responsibility | Owner |
|----------------|-------|
| Turn-by-turn dialogue | This agent (UAEP + session) |
| Tool selection loop | `CatalogToolPlanner` on `RuntimeConfig` (planned IAA-6) |
| Specialist work | **Not** this agent — Nexus delegates to mounted platform agents |
| LLM provider choice | **Not** this agent — Tier-3 `ApplicationEnvironmentProfile.llm_profile` |

---

## Capability

`platform.assist` — see `capabilities.py`.

---

## Module map

| Path | Role |
|------|------|
| `intergrax_assistant_agent.py` | `IntergraxAssistantAgent` — UAEP entry |
| `contract.py` | `AgentContract` — tools, skills, risk, max_steps |
| ``on_next_step` / cognitive pattern hooks` | Domain pipeline (replace scaffold stub) |
| `prompts/system.md` | System instructions for concierge tone |

---

## Boundaries

- Imports only `intergrax.*` and `agents/intergrax_assistant` — **no** `applications/` imports
- Must not import or call other Tier-2 agents — request delegation via Nexus plan/handoff
- Vendor SDK access only through `ToolRuntime` / integrations wired by Tier-3 host

---

## Implementation tracker

[`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md)
